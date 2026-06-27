#!/usr/bin/env python3
"""
Analyze controlled-perturbation OpenVLA/LIBERO sweep outputs.

Input:
  spike2_rollouts.csv
  spike2_trajectories/rollout_*.json

Output:
  grasp_vs_object_wide.png
  FINDINGS.md
"""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
CSV_PATH = ROOT / "spike2_rollouts.csv"
TRAJ_DIR = ROOT / "spike2_trajectories"
OUT_PLOT = ROOT / "grasp_vs_object_wide.png"
FINDINGS_PATH = ROOT / "FINDINGS.md"
PLOT_TITLE = "Wide perturbation: grasp point vs object initial position"


def configure(
    csv_path: Path,
    traj_dir: Path,
    out_plot: Path,
    label: str = "perturbation",
) -> None:
    global CSV_PATH, TRAJ_DIR, OUT_PLOT, PLOT_TITLE
    CSV_PATH = csv_path
    TRAJ_DIR = traj_dir
    OUT_PLOT = out_plot
    PLOT_TITLE = f"{label}: grasp point vs object initial position"

NARROW_SPIKE = {
    "x_slope": 0.5359,
    "x_r2": 0.5342,
    "y_slope": 0.2421,
    "y_r2": 0.0866,
    "x_span_cm": 5.29,
    "y_span_cm": 2.96,
}


def parse_bool(value: str) -> bool:
    return value.strip().lower() in {"true", "1", "yes"}


def load_trajectory(path: Path) -> np.ndarray:
    with path.open() as f:
        data = json.load(f)
    arr = np.asarray(data, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"expected trajectory shape (T, 3), got {arr.shape}")
    if arr.shape[0] == 0:
        raise ValueError("empty trajectory")
    return arr


def grasp_point(traj: np.ndarray) -> tuple[np.ndarray, int]:
    idx = int(np.where(traj[:, 2] == traj[:, 2].min())[0][0])
    return traj[idx], idx


def linear_stats(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    if len(x) < 2 or np.std(x) == 0:
        return {"slope": math.nan, "intercept": math.nan, "r2": math.nan, "pearson": math.nan}
    slope, intercept = np.polyfit(x, y, 1)
    y_hat = slope * x + intercept
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else math.nan
    pearson = float(np.corrcoef(x, y)[0, 1]) if np.std(y) > 0 else math.nan
    return {"slope": float(slope), "intercept": float(intercept), "r2": float(r2), "pearson": pearson}


def cm_span(vals: np.ndarray) -> float:
    return float(vals.max() - vals.min()) * 100.0


def read_rows() -> tuple[list[dict[str, Any]], list[tuple[str, str]]]:
    rows: list[dict[str, Any]] = []
    skipped: list[tuple[str, str]] = []

    with CSV_PATH.open(newline="") as f:
        reader = csv.DictReader(f)
        print(f"CSV columns: {', '.join(reader.fieldnames or [])}")
        print(f"Trajectory dir: {TRAJ_DIR} (pattern: rollout_*.json)\n")
        for raw in reader:
            rollout_id = raw.get("rollout_id", "?")
            if raw.get("status", "ok") != "ok":
                skipped.append((rollout_id, f"status={raw.get('status')} error={raw.get('error')}"))
                continue

            traj_path = Path(raw["ee_trajectory_path"])
            if not traj_path.is_file():
                alt = TRAJ_DIR / f"rollout_{int(rollout_id):03d}.json"
                if alt.is_file():
                    traj_path = alt
                else:
                    skipped.append((rollout_id, f"trajectory missing: {traj_path}"))
                    continue

            try:
                traj = load_trajectory(traj_path)
                gp, grasp_step = grasp_point(traj)
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                skipped.append((rollout_id, str(exc)))
                continue

            try:
                base_idx = int(raw.get("base_idx", -1))
            except (TypeError, ValueError):
                base_idx = -1

            rows.append(
                {
                    "rollout_id": int(rollout_id),
                    "grid_x": float(raw["grid_x"]),
                    "grid_y": float(raw["grid_y"]),
                    "base_idx": base_idx,
                    "obj_x": float(raw["actual_object_init_x"]),
                    "obj_y": float(raw["actual_object_init_y"]),
                    "obj_z": float(raw["actual_object_init_z"]),
                    "grasp_x": float(raw.get("grasp_point_x") or gp[0]),
                    "grasp_y": float(raw.get("grasp_point_y") or gp[1]),
                    "grasp_z": float(raw.get("grasp_point_z") or gp[2]),
                    "grasp_step": grasp_step,
                    "num_steps": int(raw["num_steps"]),
                    "success": parse_bool(raw.get("success", "")),
                    "override_error_m": float(raw.get("override_error_m") or math.nan),
                }
            )
    rows.sort(key=lambda r: r["rollout_id"])
    return rows, skipped


def interpretation(stats_x: dict[str, float], stats_y: dict[str, float], x_span: float, y_span: float) -> str:
    def axis_read(axis: str, stats: dict[str, float], span_cm: float) -> str:
        slope = stats["slope"]
        r2 = stats["r2"]
        if math.isnan(slope) or math.isnan(r2):
            return f"{axis}: inconclusive (not enough valid variation)."
        if span_cm < 10:
            return (
                f"{axis}: still narrow ({span_cm:.1f} cm), so treat slope={slope:.2f}, "
                f"R2={r2:.2f} cautiously."
            )
        if slope >= 0.75 and r2 >= 0.60:
            return f"{axis}: grasp tracks object strongly (slope={slope:.2f}, R2={r2:.2f}); grounding signal present."
        if abs(slope) <= 0.30 and r2 <= 0.20:
            return f"{axis}: grasp barely tracks object (slope={slope:.2f}, R2={r2:.2f}); replay/shortcut signal present."
        return f"{axis}: mixed partial tracking (slope={slope:.2f}, R2={r2:.2f}); pilot is not decisive on this axis."

    return axis_read("x", stats_x, x_span) + "\n" + axis_read("y", stats_y, y_span)


def write_plot(rows: list[dict[str, Any]], stats_x: dict[str, float], stats_y: dict[str, float]) -> None:
    obj_x = np.asarray([r["obj_x"] for r in rows])
    obj_y = np.asarray([r["obj_y"] for r in rows])
    grasp_x = np.asarray([r["grasp_x"] for r in rows])
    grasp_y = np.asarray([r["grasp_y"] for r in rows])

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, obj, grasp, axis_name, stats in [
        (axes[0], obj_x, grasp_x, "x", stats_x),
        (axes[1], obj_y, grasp_y, "y", stats_y),
    ]:
        colors = ["#2ca02c" if row["success"] else "#d62728" for row in rows]
        ax.scatter(obj, grasp, s=60, c=colors, zorder=3)
        for row in rows:
            ax.annotate(
                str(row["rollout_id"]),
                (row[f"obj_{axis_name}"], row[f"grasp_{axis_name}"]),
                fontsize=7,
                xytext=(3, 3),
                textcoords="offset points",
            )

        lo = min(float(obj.min()), float(grasp.min()))
        hi = max(float(obj.max()), float(grasp.max()))
        pad = 0.03 * (hi - lo + 1e-9)
        lim = (lo - pad, hi + pad)
        ax.plot(lim, lim, "k--", linewidth=1, label="y=x")
        if not math.isnan(stats["slope"]):
            xx = np.linspace(lim[0], lim[1], 50)
            ax.plot(xx, stats["slope"] * xx + stats["intercept"], "r-", linewidth=1.2, label="OLS fit")
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(f"actual object init {axis_name}")
        ax.set_ylabel(f"grasp point {axis_name}")
        ax.set_title(f"{axis_name}: slope={stats['slope']:.2f}, R2={stats['r2']:.2f}")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    fig.suptitle(PLOT_TITLE)
    fig.tight_layout()
    fig.savefig(OUT_PLOT, dpi=150)
    print(f"Saved plot to {OUT_PLOT}")


def write_findings(
    rows: list[dict[str, Any]],
    skipped: list[tuple[str, str]],
    stats_x: dict[str, float],
    stats_y: dict[str, float],
    ratio_x: float,
    ratio_y: float,
    x_span_cm: float,
    y_span_cm: float,
) -> None:
    read = interpretation(stats_x, stats_y, x_span_cm, y_span_cm)
    success_count = sum(1 for row in rows if row["success"])
    content = f"""# Controlled-Perturbation Pilot Findings

## Setup

- Input CSV: `{CSV_PATH.name}`
- Trajectories: `{TRAJ_DIR.name}/rollout_*.json`
- Grasp definition: EE `[x, y, z]` at the global minimum-z timestep (earliest if tied)
- Valid rollouts used in regression: {len(rows)}
- Successful rollouts among valid rows: {success_count}/{len(rows)}
- Skipped rollouts: {len(skipped)}

## Position Spread

| Axis | Wide pilot object span | Narrow spike object span |
|------|------------------------|--------------------------|
| x | {x_span_cm:.2f} cm | {NARROW_SPIKE["x_span_cm"]:.2f} cm |
| y | {y_span_cm:.2f} cm | {NARROW_SPIKE["y_span_cm"]:.2f} cm |

## Regression: grasp_point ~ actual_object_init

| Axis | Slope | R2 | Pearson r | Tracking ratio |
|------|-------|----|-----------|----------------|
| x | {stats_x["slope"]:.4f} | {stats_x["r2"]:.4f} | {stats_x["pearson"]:.4f} | {ratio_x:.4f} |
| y | {stats_y["slope"]:.4f} | {stats_y["r2"]:.4f} | {stats_y["pearson"]:.4f} | {ratio_y:.4f} |

## Read

{read}

Compared to the narrow spike (`x`: slope={NARROW_SPIKE["x_slope"]:.2f}, R2={NARROW_SPIKE["x_r2"]:.2f}; `y`: slope={NARROW_SPIKE["y_slope"]:.2f}, R2={NARROW_SPIKE["y_r2"]:.2f}), this wide perturbation run is the first real test of whether the detector separates grounding from replay outside the natural 3-5 cm variation.

## Honesty Notes

- This is still a pilot, not proof.
- Regression uses all valid completed rollouts because only {success_count} succeeded; a successes-only fit would be too underpowered for this pilot.
- If many extreme perturbations fail or clip into scene objects, interpret the usable range rather than the commanded range.
"""
    FINDINGS_PATH.write_text(content)
    print(f"Wrote findings to {FINDINGS_PATH}")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Analyze perturbation sweep outputs")
    parser.add_argument(
        "--dataset",
        choices=("spike2", "spike3"),
        default="spike2",
        help="Which sweep outputs to analyze (default: spike2)",
    )
    args = parser.parse_args()

    if args.dataset == "spike3":
        configure(
            ROOT / "spike3_rollouts.csv",
            ROOT / "spike3_trajectories",
            ROOT / "spike3_grasp_vs_object.png",
            label="Tight-grid perturbation",
        )

    if not CSV_PATH.exists():
        sys.exit(f"Missing {CSV_PATH}. Run modal_perturbation_sweep.py first.")

    rows, skipped = read_rows()
    if skipped:
        print("Skipped rollouts:")
        for rollout_id, reason in skipped:
            print(f"  rollout_id={rollout_id}: {reason}")
        print()

    if len(rows) < 2:
        sys.exit("Need at least two successful rollouts with valid trajectories for regression.")

    print(
        "Grasp point definition: EE [x,y,z] at the global minimum-z timestep "
        "(earliest if tied).\n"
    )
    success_count = sum(1 for row in rows if row["success"])
    print(f"Valid rollouts used: {len(rows)} (successes: {success_count})\n")
    print(
        f"{'id':>3} {'obj_x':>8} {'obj_y':>8} {'grasp_x':>8} "
        f"{'grasp_y':>8} {'grasp_z':>8} {'ok':>5} {'steps':>5}"
    )
    print("-" * 70)
    for row in rows:
        print(
            f"{row['rollout_id']:3d} {row['obj_x']:8.4f} {row['obj_y']:8.4f} "
            f"{row['grasp_x']:8.4f} {row['grasp_y']:8.4f} {row['grasp_z']:8.4f} "
            f"{str(row['success']):>5} {row['num_steps']:5d}"
        )
    print()

    obj_x = np.asarray([r["obj_x"] for r in rows])
    obj_y = np.asarray([r["obj_y"] for r in rows])
    grasp_x = np.asarray([r["grasp_x"] for r in rows])
    grasp_y = np.asarray([r["grasp_y"] for r in rows])

    stats_x = linear_stats(obj_x, grasp_x)
    stats_y = linear_stats(obj_y, grasp_y)
    x_span_cm = cm_span(obj_x)
    y_span_cm = cm_span(obj_y)
    grasp_x_span_cm = cm_span(grasp_x)
    grasp_y_span_cm = cm_span(grasp_y)
    ratio_x = grasp_x_span_cm / x_span_cm if x_span_cm > 0 else math.nan
    ratio_y = grasp_y_span_cm / y_span_cm if y_span_cm > 0 else math.nan

    print("Ranges:")
    print(f"  object_init_x span: {x_span_cm:.2f} cm")
    print(f"  object_init_y span: {y_span_cm:.2f} cm")
    print(f"  grasp_x span:       {grasp_x_span_cm:.2f} cm")
    print(f"  grasp_y span:       {grasp_y_span_cm:.2f} cm\n")

    print("Linear fit: grasp ~ actual_object_init")
    print(
        f"  x: slope={stats_x['slope']:.4f}, intercept={stats_x['intercept']:.4f}, "
        f"R2={stats_x['r2']:.4f}, Pearson r={stats_x['pearson']:.4f}"
    )
    print(
        f"  y: slope={stats_y['slope']:.4f}, intercept={stats_y['intercept']:.4f}, "
        f"R2={stats_y['r2']:.4f}, Pearson r={stats_y['pearson']:.4f}\n"
    )

    print("Tracking ratio = grasp range / object init range")
    print(f"  x: {ratio_x:.4f}")
    print(f"  y: {ratio_y:.4f}\n")

    print("Plain-English read:")
    print(interpretation(stats_x, stats_y, x_span_cm, y_span_cm))
    print()

    write_plot(rows, stats_x, stats_y)
    write_findings(rows, skipped, stats_x, stats_y, ratio_x, ratio_y, x_span_cm, y_span_cm)


if __name__ == "__main__":
    main()
