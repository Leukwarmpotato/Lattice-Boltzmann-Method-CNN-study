import json
import os
import time

import numpy as np

from LBM_PM_calc import calculate_permeability

FORCE = 1e-6


def channel_case(width, height, nu):
    mask = np.ones((height, height), dtype=bool)
    mask[:, width:] = False
    k, _, steps, converged, _ = calculate_permeability(mask, nu, FORCE, max_steps=80000, min_steps=5000, tol=1e-7)
    exact = width**3 / (12 * height)
    return k, exact, steps, converged


def sangani_acrivos(solid_fraction, radius):
    c = solid_fraction
    return radius**2 / (8 * c) * (-np.log(c) - 1.47633597 + 2 * c - 1.77428264 * c**2 + 4.07770444 * c**3)


def cylinder_case(radius, size=100, nu=1 / 6):
    x = np.arange(size) + 0.5
    X, Y = np.meshgrid(x, x, indexing="ij")
    mask = (X - size / 2) ** 2 + (Y - size / 2) ** 2 > radius**2
    k, _, steps, converged, _ = calculate_permeability(mask, nu, FORCE, max_steps=150000, min_steps=5000, tol=1e-5)
    solid_fraction = 1 - mask.mean()
    effective_radius = np.sqrt(solid_fraction * size**2 / np.pi)
    exact = sangani_acrivos(solid_fraction, effective_radius)
    return k, exact, solid_fraction, steps, converged


def main():
    rows = []
    start = time.time()

    for width, height in ((20, 30), (8, 12)):
        for nu in (0.01, 0.05, 1 / 6):
            k, exact, steps, converged = channel_case(width, height, nu)
            rows.append({
                "case": f"Channel, width {width}, nu {nu:.3f}",
                "lbm": float(k),
                "analytical": float(exact),
                "error_percent": float(100 * (k / exact - 1)),
                "steps": steps,
                "converged": converged,
            })

    for radius in (10, 20):
        k, exact, solid_fraction, steps, converged = cylinder_case(radius)
        rows.append({
            "case": f"Cylinder array, radius {radius}, solid fraction {solid_fraction:.3f}",
            "lbm": float(k),
            "analytical": float(exact),
            "error_percent": float(100 * (k / exact - 1)),
            "steps": steps,
            "converged": converged,
        })

    os.makedirs("results", exist_ok=True)
    with open("results/validation.json", "w") as f:
        json.dump(rows, f, indent=2)

    print("\nSolver validation against analytical solutions")
    print(f"{'Case':52s} {'LBM k':>10s} {'Exact k':>10s} {'Error':>9s}")
    for row in rows:
        flag = "" if row["converged"] else "  (step limit)"
        print(f"{row['case']:52s} {row['lbm']:10.4f} {row['analytical']:10.4f} {row['error_percent']:+8.2f}%{flag}")
    print(f"\nFinished in {time.time() - start:.0f}s, saved results/validation.json")


if __name__ == "__main__":
    main()
