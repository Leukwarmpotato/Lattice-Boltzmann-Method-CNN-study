import argparse
import copy
import json
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from matplotlib.ticker import FixedLocator, NullLocator, ScalarFormatter
from scipy.optimize import minimize
from scipy.stats import norm, spearmanr


def load_data(path):
    data = np.load(path, allow_pickle=True)
    X = data["X"].astype(np.float32)
    k = data["y"].astype(np.float64)
    porosity = X.reshape(len(X), -1).mean(axis=1).astype(np.float64)
    return X, k, porosity


def split_indices(n, seed=42, test_size=0.15, val_size=0.15):
    idx = np.arange(n)
    train_val, test = train_test_split(idx, test_size=test_size, random_state=seed)
    train, val = train_test_split(train_val, test_size=val_size / (1 - test_size), random_state=seed)
    return train, val, test


def kozeny_carman_feature(porosity):
    return np.log(porosity**3 / (1 - porosity) ** 2)


class KozenyCarman:
    def fit(self, porosity, k):
        A = np.vstack([np.ones_like(porosity), kozeny_carman_feature(porosity)]).T
        self.coef = np.linalg.lstsq(A, np.log(k), rcond=None)[0]
        return self

    def predict_log(self, porosity):
        return self.coef[0] + self.coef[1] * kozeny_carman_feature(porosity)


class PeriodicCNN(nn.Module):
    def __init__(self, width=32, dropout=0.2):
        super().__init__()

        def block(c_in, c_out):
            return nn.Sequential(
                nn.Conv2d(c_in, c_out, 3, padding=1, padding_mode="circular"),
                nn.BatchNorm2d(c_out),
                nn.ReLU(),
                nn.Conv2d(c_out, c_out, 3, padding=1, padding_mode="circular"),
                nn.BatchNorm2d(c_out),
                nn.ReLU(),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(block(1, width), block(width, 2 * width), block(2 * width, 4 * width))
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(8 * width, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        h = self.features(x)
        pooled = torch.cat([h.mean(dim=(2, 3)), h.amax(dim=(2, 3))], dim=1)
        return self.head(pooled).squeeze(1)


def augment(x):
    n, _, h, w = x.shape
    shift_x = int(torch.randint(0, h, (1,)))
    shift_y = int(torch.randint(0, w, (1,)))
    x = torch.roll(x, shifts=(shift_x, shift_y), dims=(2, 3))
    if torch.rand(1) < 0.5:
        x = torch.flip(x, dims=(2,))
    if torch.rand(1) < 0.5:
        x = torch.flip(x, dims=(3,))
    return x


def train_cnn(X_train, t_train, X_val, t_val, seed, epochs, patience, augment_data, device):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = PeriodicCNN().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=8)
    loss_fn = nn.MSELoss()

    Xt = torch.tensor(X_train[:, None]).to(device)
    tt = torch.tensor(t_train, dtype=torch.float32).to(device)
    Xv = torch.tensor(X_val[:, None]).to(device)
    tv = torch.tensor(t_val, dtype=torch.float32).to(device)

    best_loss, best_state, wait = np.inf, None, 0
    generator = torch.Generator().manual_seed(seed)
    for epoch in range(epochs):
        model.train()
        order = torch.randperm(len(Xt), generator=generator)
        for start in range(0, len(Xt), 32):
            batch = order[start:start + 32]
            xb = Xt[batch]
            if augment_data:
                xb = augment(xb)
            optimizer.zero_grad()
            loss = loss_fn(model(xb), tt[batch])
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(Xv), tv).item()
        scheduler.step(val_loss)
        if val_loss < best_loss:
            best_loss, best_state, wait = val_loss, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if wait >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    return model, epoch + 1


def predict(model, X, device, symmetrise=True):
    with torch.no_grad():
        x = torch.tensor(X[:, None]).to(device)
        views = [x]
        if symmetrise:
            views += [torch.flip(x, dims=(2,)), torch.flip(x, dims=(3,)), torch.flip(x, dims=(2, 3))]
        return torch.stack([model(v) for v in views]).mean(0).cpu().numpy().astype(np.float64)


def metrics(k_true, log_k_pred):
    k_pred = np.exp(log_k_pred)
    log_true = np.log(k_true)
    r2 = 1 - np.sum((k_true - k_pred) ** 2) / np.sum((k_true - k_true.mean()) ** 2)
    r2_log = 1 - np.sum((log_true - log_k_pred) ** 2) / np.sum((log_true - log_true.mean()) ** 2)
    return {
        "r2": float(r2),
        "r2_log": float(r2_log),
        "rmse": float(np.sqrt(np.mean((k_true - k_pred) ** 2))),
        "mape": float(np.mean(np.abs(k_pred - k_true) / k_true) * 100),
    }


def summarise(runs):
    keys = runs[0].keys()
    return {key: {"mean": float(np.mean([r[key] for r in runs])), "std": float(np.std([r[key] for r in runs]))} for key in keys}


def run_variant(name, X, k, porosity, train, val, test, kc, seeds, epochs, patience, augment_data, residual, device):
    base = kc.predict_log(porosity) if residual else np.zeros_like(porosity)
    target = np.log(k) - base
    mu, sigma = target[train].mean(), target[train].std()
    scaled = (target - mu) / sigma

    runs, val_preds, test_preds = [], [], []
    for seed in seeds:
        start = time.time()
        model, n_epochs = train_cnn(X[train], scaled[train], X[val], scaled[val], seed, epochs, patience, augment_data, device)
        val_log = base[val] + mu + sigma * predict(model, X[val], device)
        test_log = base[test] + mu + sigma * predict(model, X[test], device)
        m = metrics(k[test], test_log)
        runs.append(m)
        val_preds.append(val_log)
        test_preds.append(test_log)
        print(f"{name}, seed {seed}: test R2 {m['r2']:.4f}, MAPE {m['mape']:.1f}% ({n_epochs} epochs, {time.time() - start:.0f}s)")

    return runs, np.array(val_preds), np.array(test_preds)


def calibrate(val_mean, val_std, k_val):
    residual = np.log(k_val) - val_mean

    def nll(params):
        variance = np.exp(params[0]) * val_std**2 + np.exp(params[1])
        return np.mean(0.5 * np.log(variance) + 0.5 * residual**2 / variance)

    best = minimize(nll, x0=[0.0, np.log(np.var(residual))], method="Nelder-Mead")
    return float(np.exp(best.x[0])), float(np.exp(best.x[1]))


def total_std(ens_std, a, b):
    return np.sqrt(a * ens_std**2 + b)


def coverage_curve(mean, std, k_true):
    levels = np.linspace(0.05, 0.95, 19)
    z = np.abs(np.log(k_true) - mean) / std
    observed = [float(np.mean(z <= norm.ppf(0.5 + p / 2))) for p in levels]
    return levels, np.array(observed)


def plot_parity(k_test, kc_log, ens_log, ens_std, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharex=True, sharey=True)
    lims = [k_test.min() * 0.8, k_test.max() * 1.25]
    for ax, pred, title in ((axes[0], kc_log, "Kozeny-Carman (porosity only)"), (axes[1], ens_log, "Residual CNN ensemble")):
        if pred is ens_log:
            lower, upper = np.exp(pred - 1.96 * ens_std), np.exp(pred + 1.96 * ens_std)
            ax.errorbar(k_test, np.exp(pred), yerr=[np.exp(pred) - lower, upper - np.exp(pred)], fmt="o", ms=4, alpha=0.6, elinewidth=0.8, capsize=0)
        else:
            ax.scatter(k_test, np.exp(pred), s=16, alpha=0.6)
        ax.plot(lims, lims, "k--", lw=1)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ticks = [t for t in (2, 5, 10, 20, 50, 100, 200) if lims[0] <= t <= lims[1]]
        for axis in (ax.xaxis, ax.yaxis):
            axis.set_major_locator(FixedLocator(ticks))
            axis.set_minor_locator(NullLocator())
            axis.set_major_formatter(ScalarFormatter())
        m = metrics(k_test, pred)
        ax.set_title(f"{title}\nR² = {m['r2']:.3f}, MAPE = {m['mape']:.1f}%")
        ax.set_xlabel("LBM permeability (lattice units)")
        ax.grid(True, alpha=0.3, which="both")
    axes[0].set_ylabel("Predicted permeability (lattice units)")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_error_vs_porosity(porosity_test, k_test, kc_log, ens_log, path):
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.scatter(porosity_test, 100 * (np.exp(kc_log) - k_test) / k_test, s=16, alpha=0.6, label="Kozeny-Carman")
    ax.scatter(porosity_test, 100 * (np.exp(ens_log) - k_test) / k_test, s=16, alpha=0.6, label="Residual CNN ensemble")
    ax.axhline(0, color="k", lw=1)
    ax.set_xlabel("Porosity")
    ax.set_ylabel("Prediction error (%)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_calibration(levels, observed, ens_std, abs_err, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot(levels, observed, "o-", label="Ensemble (calibrated)")
    axes[0].plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")
    axes[0].set_xlabel("Expected coverage")
    axes[0].set_ylabel("Observed coverage on test set")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[1].scatter(ens_std, abs_err, s=16, alpha=0.6)
    axes[1].set_xlabel("Ensemble standard deviation of log k")
    axes[1].set_ylabel("Absolute error in log k")
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_learning_curve(sizes, cnn_mape, kc_mape, path):
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    mean = np.array([np.mean(v) for v in cnn_mape])
    std = np.array([np.std(v) for v in cnn_mape])
    ax.errorbar(sizes, mean, yerr=std, fmt="o-", capsize=3, label="Residual CNN")
    ax.plot(sizes, kc_mape, "s--", label="Kozeny-Carman")
    ax.set_xlabel("Number of training samples")
    ax.set_ylabel("Test MAPE (%)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate permeability surrogates")
    parser.add_argument("--data", default="nn_dataset.npz")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--learning-curve", action="store_true")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    if args.quick:
        args.seeds, args.epochs, args.patience = 2, 15, 5

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs("figures", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    X, k, porosity = load_data(args.data)
    train, val, test = split_indices(len(k))
    print(f"{len(k)} samples: {len(train)} train, {len(val)} validation, {len(test)} test")
    seeds = list(range(args.seeds))

    kc = KozenyCarman().fit(porosity[train], k[train])
    kc_test = kc.predict_log(porosity[test])
    results = {"kozeny_carman": metrics(k[test], kc_test)}
    print(f"Kozeny-Carman: test R2 {results['kozeny_carman']['r2']:.4f}, MAPE {results['kozeny_carman']['mape']:.1f}%")

    variants = {
        "cnn_direct": dict(residual=False, augment_data=True),
        "cnn_residual_no_augmentation": dict(residual=True, augment_data=False),
        "cnn_residual": dict(residual=True, augment_data=True),
    }
    ensemble = {}
    for name, options in variants.items():
        runs, val_preds, test_preds = run_variant(name, X, k, porosity, train, val, test, kc, seeds, args.epochs, args.patience, device=device, **options)
        results[name] = summarise(runs)
        results[name + "_ensemble"] = metrics(k[test], test_preds.mean(0))
        ensemble[name] = (val_preds, test_preds)

    val_preds, test_preds = ensemble["cnn_residual"]
    val_mean, val_std = val_preds.mean(0), val_preds.std(0) + 1e-6
    test_mean, test_std = test_preds.mean(0), test_preds.std(0) + 1e-6
    a, b = calibrate(val_mean, val_std, k[val])
    calibrated_std = total_std(test_std, a, b)
    abs_err = np.abs(np.log(k[test]) - test_mean)
    levels, observed_raw = coverage_curve(test_mean, test_std, k[test])
    levels, observed = coverage_curve(test_mean, calibrated_std, k[test])
    results["uncertainty"] = {
        "ensemble_variance_weight": a,
        "noise_variance": b,
        "coverage_95_uncalibrated": float(observed_raw[-1]),
        "coverage_95_calibrated": float(observed[-1]),
        "spearman_ensemble_std_vs_error": float(spearmanr(test_std, abs_err)[0]),
    }

    plot_parity(k[test], kc_test, test_mean, calibrated_std, "figures/parity_kc_vs_cnn.png")
    plot_error_vs_porosity(porosity[test], k[test], kc_test, test_mean, "figures/error_vs_porosity.png")
    plot_calibration(levels, observed, test_std, abs_err, "figures/uncertainty_calibration.png")

    if args.learning_curve:
        sizes = [100, 200, 400, len(train)]
        cnn_mape, kc_mape = [], []
        rng = np.random.default_rng(0)
        for size in sizes:
            subset = np.sort(rng.choice(train, size=size, replace=False)) if size < len(train) else train
            kc_sub = KozenyCarman().fit(porosity[subset], k[subset])
            kc_mape.append(metrics(k[test], kc_sub.predict_log(porosity[test]))["mape"])
            runs, _, _ = run_variant(f"learning curve n={size}", X, k, porosity, subset, val, test, kc_sub, seeds[:3], args.epochs, args.patience, augment_data=True, residual=True, device=device)
            cnn_mape.append([r["mape"] for r in runs])
        results["learning_curve"] = {"sizes": sizes, "cnn_mape": cnn_mape, "kozeny_carman_mape": kc_mape}
        plot_learning_curve(sizes, cnn_mape, kc_mape, "figures/learning_curve.png")

    with open("results/metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nTest set results")
    print(f"{'Model':34s} {'R2':>14s} {'MAPE (%)':>14s}")
    print(f"{'Kozeny-Carman':34s} {results['kozeny_carman']['r2']:14.4f} {results['kozeny_carman']['mape']:14.1f}")
    for name in variants:
        s = results[name]
        print(f"{name:34s} {s['r2']['mean']:8.4f} ± {s['r2']['std']:.3f} {s['mape']['mean']:8.1f} ± {s['mape']['std']:.1f}")
        e = results[name + "_ensemble"]
        print(f"{name + ' (ensemble)':34s} {e['r2']:14.4f} {e['mape']:14.1f}")
    u = results["uncertainty"]
    print(f"\n95% interval coverage: {100 * u['coverage_95_uncalibrated']:.0f}% from ensemble spread alone, {100 * u['coverage_95_calibrated']:.0f}% after calibration")
    print(f"Rank correlation between ensemble spread and error: {u['spearman_ensemble_std_vs_error']:.2f}")
    print("Saved results/metrics.json and figures/")


if __name__ == "__main__":
    main()
