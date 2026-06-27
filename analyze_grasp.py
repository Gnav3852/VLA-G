#!/usr/bin/env python3
"""
Analyze whether OpenVLA grasp points track bowl initial position across rollouts.

Reads spike_rollouts.csv + spike_trajectories/seed_*.json only. No model/sim.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
CSV_PATH = ROOT / "spike_rollouts.csv"
TRAJ_DIR = ROOT / "spike_trajectories"
OUT_PLOT = ROOT / "grasp_vs_object.png"

# Grasp point: EE [x,y,z] at the global minimum-z timestep (bottom of reach toward bowl).
# If multiple timesteps tie at min-z, use the earliest.


def parse_bool(val: str) -> bool:
    return val.strip().lower() in {"true", "1", "yes"}


def load_trajectory(path: Path) -> np.ndarray:
    with open(path) as f:
        data = json.load(f)
    arr = np.asarray(data, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"expected (T, 3) trajectory, got shape {arr.shape}")
    if arr.shape[0] == 0:
        raise ValueError("empty trajectory")
    return arr


def grasp_point(traj: np.ndarray) -> tuple[np.ndarray, int]:
    z = traj[:, 2]
    min_z = z.min()
    idx = int(np.where(z == min_z)[0][0])  # earliest if tied
    return traj[idx], idx


def linear_stats(x: np.ndarray, y: np.ndarray) -> dict:
    slope, intercept = np.polyfit(x, y, 1)
    y_hat = slope * x + intercept
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    pearson = float(np.corrcoef(x, y)[0, 1]) if len(x) > 1 and np.std(x) > 0 and np.std(y) > 0 else float("nan")
    return {"slope": slope, "intercept": intercept, "r2": r2, "pearson": pearson}


def interpret(slope: float, r2: float, axis: str) -> str:
    if np.isnan(r2) or np.isnan(slope):
        return f"{axis}-axis: inconclusive (insufficient variation or too few points)."
    abs_r2 = abs(r2)
    if abs_r2 < 0.15:
        return (
            f"{axis}-axis slope = {slope:.2f}, R² = {r2:.2f} → inconclusive — "
            f"position variation may be too small to separate grounding vs replay."
        )
    if abs(slope) >= 0.6 and r2 >= 0.4:
        return (
            f"{axis}-axis slope = {slope:.2f}, R² = {r2:.2f} → grasp point tracks bowl "
            f"{axis}-position. Consistent with grounding."
        )
    if abs(slope) <= 0.3 and r2 < 0.4:
        return (
            f"{axis}-axis slope = {slope:.2f}, R² = {r2:.2f} → grasp point is nearly "
            f"independent of bowl {axis}-position. Consistent with reaching to a "
            f"memorized spot, NOT grounding."
        )
    return (
        f"{axis}-axis slope = {slope:.2f}, R² = {r2:.2f} → weak/mixed signal; "
        f"neither strong grounding nor clear replay signature."
    )


def main() -> None:
    if not CSV_PATH.exists():
        sys.exit(f"Missing {CSV_PATH}")

    rows = []
    skipped = []

    with open(CSV_PATH, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        print(f"CSV columns: {', '.join(fieldnames)}")
        print(f"Trajectory dir: {TRAJ_DIR} (pattern: seed_*.json)\n")

        for row in reader:
            seed = row.get("seed", "?")
            if not parse_bool(row.get("success", "")):
                skipped.append((seed, "not successful"))
                continue

            traj_path = Path(row["ee_trajectory_path"])
            if not traj_path.is_file():
                alt = TRAJ_DIR / f"seed_{seed}.json"
                if alt.is_file():
                    traj_path = alt
                else:
                    skipped.append((seed, f"trajectory missing: {row['ee_trajectory_path']}"))
                    continue

            try:
                traj = load_trajectory(traj_path)
                gp, grasp_step = grasp_point(traj)
            except (ValueError, json.JSONDecodeError, OSError) as e:
                skipped.append((seed, str(e)))
                continue

            rows.append(
                {
                    "seed": int(seed),
                    "obj_x": float(row["target_object_init_x"]),
                    "obj_y": float(row["target_object_init_y"]),
                    "obj_z": float(row["target_object_init_z"]),
                    "grasp_x": float(gp[0]),
                    "grasp_y": float(gp[1]),
                    "grasp_z": float(gp[2]),
                    "traj_len": traj.shape[0],
                    "grasp_step": grasp_step,
                }
            )

    print(
        "Grasp point definition: EE [x,y,z] at the global minimum-z timestep "
        "(earliest if tied).\n"
    )

    if skipped:
        print("Skipped rollouts:")
        for seed, reason in skipped:
            print(f"  seed={seed}: {reason}")
        print()

    n = len(rows)
    print(f"Rollouts used in analysis: {n}\n")
    if n < 2:
        sys.exit("Need at least 2 successful rollouts with valid trajectories.")

    rows.sort(key=lambda r: r["seed"])

    print(f"{'seed':>4}  {'obj_x':>8} {'obj_y':>8}  {'grasp_x':>8} {'grasp_y':>8}  {'grasp_z':>8}  step")
    print("-" * 62)
    for r in rows:
        print(
            f"{r['seed']:4d}  {r['obj_x']:8.4f} {r['obj_y']:8.4f}  "
            f"{r['grasp_x']:8.4f} {r['grasp_y']:8.4f}  {r['grasp_z']:8.4f}  {r['grasp_step']:4d}"
        )
    print()

    obj_x = np.array([r["obj_x"] for r in rows])
    obj_y = np.array([r["obj_y"] for r in rows])
    grasp_x = np.array([r["grasp_x"] for r in rows])
    grasp_y = np.array([r["grasp_y"] for r in rows])

    stats_x = linear_stats(obj_x, grasp_x)
    stats_y = linear_stats(obj_y, grasp_y)

    print("Linear fit: grasp ~ object_init")
    print(
        f"  x: slope={stats_x['slope']:.4f}, intercept={stats_x['intercept']:.4f}, "
        f"R²={stats_x['r2']:.4f}, Pearson r={stats_x['pearson']:.4f}"
    )
    print(
        f"  y: slope={stats_y['slope']:.4f}, intercept={stats_y['intercept']:.4f}, "
        f"R²={stats_y['r2']:.4f}, Pearson r={stats_y['pearson']:.4f}"
    )
    print()
    print("Read:")
    print(f"  {interpret(stats_x['slope'], stats_x['r2'], 'x')}")
    print(f"  {interpret(stats_y['slope'], stats_y['r2'], 'y')}")
    print()

    obj_x_span_cm = (obj_x.max() - obj_x.min()) * 100
    obj_y_span_cm = (obj_y.max() - obj_y.min()) * 100
    print(
        f"Object init spread: x={obj_x_span_cm:.2f} cm, y={obj_y_span_cm:.2f} cm "
        f"(n={n} successful rollouts)"
    )

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    for ax, obj, grasp, label, stats in [
        (axes[0], obj_x, grasp_x, "x", stats_x),
        (axes[1], obj_y, grasp_y, "y", stats_y),
    ]:
        ax.scatter(obj, grasp, c="steelblue", s=60, zorder=3)
        for r in rows:
            ax.annotate(str(r["seed"]), (r[f"obj_{label}"], r[f"grasp_{label}"]), fontsize=7, xytext=(3, 3), textcoords="offset points")

        lo = min(obj.min(), grasp.min())
        hi = max(obj.max(), grasp.max())
        pad = 0.01 * (hi - lo + 1e-9)
        lim = (lo - pad, hi + pad)
        ax.plot(lim, lim, "k--", linewidth=1, label="y=x (perfect grounding)")
        ax.set_xlim(lim)
        ax.set_ylim(lim)

        xx = np.linspace(lim[0], lim[1], 50)
        ax.plot(xx, stats["slope"] * xx + stats["intercept"], "r-", linewidth=1.2, label="OLS fit")

        ax.set_xlabel(f"object init {label}")
        ax.set_ylabel(f"grasp point {label}")
        ax.set_title(f"{label}: slope={stats['slope']:.2f}, R²={stats['r2']:.2f}")
        ax.set_aspect("equal", adjustable="box")
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Grasp point vs bowl initial position (successful rollouts only)", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT_PLOT, dpi=150)
    print(f"Saved plot to {OUT_PLOT}")


if __name__ == "__main__":
    main()
