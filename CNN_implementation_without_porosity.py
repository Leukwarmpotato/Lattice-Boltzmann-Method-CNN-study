import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import json
import time
import copy
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

class PermeabilityCNN1D(nn.Module):
    def __init__(self, bitstring_len):
        super().__init__()

        self.conv1 = nn.Conv1d(1, 8, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(8, 16, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(16, 32, kernel_size=5, padding=2)

        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(0.2)

        conv_output_len = bitstring_len // 8

        self.fc1 = nn.Linear(32 * conv_output_len, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        return self.fc3(x)


def load_dataset(filename="nn_dataset.npz"):
    print("Loading dataset...")
    data = np.load(filename, allow_pickle=True)

    X = data["X"]          
    y = data["y"]          
    metadata = json.loads(str(data["metadata"]))

    print(f"Samples: {len(X)}")
    print(f"Input grid: {X.shape[1]} x {X.shape[2]}")
    print(f"Permeability range: {y.min():.4f} – {y.max():.4f}")

    return X, y, metadata

def prepare_data(X, y, test_size=0.15, val_size=0.15, random_state=42):

    n_samples, h, w = X.shape
    bitstring_len = h * w

    X_bitstring = X.reshape(n_samples, bitstring_len).astype(np.float32)
    y = y.astype(np.float32)

    X_train, X_test, y_train, y_test = train_test_split(
        X_bitstring, y, test_size=test_size, random_state=random_state
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=val_size / (1 - test_size), random_state=random_state
    )

    print("\nPrepared bitstrings:")
    print(f"Flattened shape: {X_bitstring.shape}")
    print(f"Bitstring length: {bitstring_len}")
    print(f"Train/Val/Test: {len(X_train)}/{len(X_val)}/{len(X_test)}")
    return X_train, X_val, X_test, y_train, y_val, y_test, bitstring_len


def train_cnn(X_train, y_train, X_val, y_val, X_test, bitstring_len,
              epochs=100, batch_size=32, patience=20):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    X_train_t = torch.FloatTensor(X_train).reshape(-1, 1, bitstring_len)
    X_val_t   = torch.FloatTensor(X_val).reshape(-1, 1, bitstring_len)
    X_test_t  = torch.FloatTensor(X_test).reshape(-1, 1, bitstring_len)
    y_train_t = torch.FloatTensor(y_train).reshape(-1, 1)
    y_val_t   = torch.FloatTensor(y_val).reshape(-1, 1)

    train_loader = DataLoader(
        TensorDataset(X_train_t, y_train_t),
        batch_size=batch_size,
        shuffle=True
    )

    model = PermeabilityCNN1D(bitstring_len).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    best_loss = float("inf")
    best_state = None
    patience_counter = 0

    val_history = []   

    for epoch in range(epochs):
        model.train()
        losses = []

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        train_loss = np.mean(losses)

        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t.to(device))
            val_loss = criterion(val_pred, y_val_t.to(device)).item()

        val_history.append(val_loss)  

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}  Train={train_loss:.4f}  Val={val_loss:.4f}")

        if val_loss < best_loss:
            best_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break

    model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        y_train_pred = model(X_train_t.to(device)).cpu().numpy().flatten()
        y_test_pred  = model(X_test_t.to(device)).cpu().numpy().flatten()

    return model, y_train_pred, y_test_pred, val_history   

def plot_all_results(y_train, y_train_pred, y_test, y_test_pred, val_history):

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

   
    axes[0].scatter(y_train, y_train_pred, s=20, alpha=0.5, label="Train")
    axes[0].scatter(y_test, y_test_pred, s=20, alpha=0.5, label="Test", color="green")

    min_val = min(y_train.min(), y_test.min())
    max_val = max(y_train.max(), y_test.max())
    axes[0].plot([min_val, max_val], [min_val, max_val], 'r--')

    axes[0].set_title("Regression: Actual vs Predicted")
    axes[0].set_xlabel("Actual")
    axes[0].set_ylabel("Predicted")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

   
    train_res = y_train - y_train_pred
    test_res = y_test - y_test_pred

    axes[1].scatter(y_train_pred, train_res, alpha=0.5, s=20, label="Train")
    axes[1].scatter(y_test_pred, test_res, alpha=0.5, s=20, label="Test", color="green")

    axes[1].axhline(0, color="red", linestyle="--")
    axes[1].set_title("Residuals Plot")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("Actual - Predicted")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    
    axes[2].plot(val_history, linewidth=2)
    axes[2].set_title("Validation Loss")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("MSE Loss")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs("figures", exist_ok=True)
    plt.savefig("figures/cnn_results.png", dpi=150, bbox_inches="tight")
    plt.close()

    
    plt.figure(figsize=(6,4))
    plt.hist(test_res, bins=25, alpha=0.75)
    plt.title("Residual Histogram (Test Set)")
    plt.xlabel("Error (Actual - Predicted)")
    plt.ylabel("Count")
    plt.grid(True, alpha=0.3)
    plt.savefig("figures/cnn_residual_histogram.png", dpi=150, bbox_inches="tight")
    plt.close()

def evaluate(y_train, y_train_pred, y_test, y_test_pred):
    print("\nEvaluation:")
    r2t = r2_score(y_train, y_train_pred)
    r2  = r2_score(y_test, y_test_pred)
    mae = mean_absolute_error(y_test, y_test_pred)
    mape = np.mean(np.abs((y_test - y_test_pred) / y_test)) * 100
    print(f"Train R²: {r2t:.4f}")
    print(f"Test  R²: {r2:.4f}")
    print(f"MAE:      {mae:.4f}")
    print(f"MAPE:     {mape:.2f}%")
    return {"train_r2": r2t, "test_r2": r2, "mae": mae, "mape": mape}

if __name__ == "__main__":

    torch.manual_seed(42)

    X, y, metadata = load_dataset()

    X_train, X_val, X_test, y_train, y_val, y_test, bit_len = prepare_data(X, y)

    model, y_train_pred, y_test_pred, val_history = train_cnn(
        X_train, y_train,
        X_val, y_val,
        X_test,
        bit_len,
        epochs=500
    )

    evaluate(y_train, y_train_pred, y_test, y_test_pred)

    plot_all_results(y_train, y_train_pred, y_test, y_test_pred, val_history)

    torch.save(model.state_dict(), "cnn_model_bitstring_only.pth")
    print("\nSaved model to cnn_model_bitstring_only.pth")
