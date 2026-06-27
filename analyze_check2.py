#!/usr/bin/env python3
"""
Analysis 2: Harden (or kill) the "grounds on x, replays on y" finding.

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
OUT_PLOT = ROOT / "displacement_vs_success.png"

# Same grasp definition as analyze_grasp.py: global minimum-z (earliest if tied).


def parse_bool(val: str) -> bool:
    return val.strip().lower() in {"true", "1", "yes"}


def load_trajectory(path: Path) -> np.ndarray:
    with open(path) as f:
        data = json.load(f)
    arr = np.asarray(data, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 3 or arr.shape[0] == 0:
        raise ValueError(f"bad trajectory shape {arr.shape}")
    return arr


def grasp_point(traj: np.ndarray) -> np.ndarray:
    z = traj[:, 2]
    idx = int(np.where(z == z.min())[0][0])
    return traj[idx]


def linear_stats(x: np.ndarray, y: np.ndarray) -> dict:
    slope, intercept = np.polyfit(x, y, 1)
    y_hat = slope * x + intercept
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    pearson = (
        float(np.corrcoef(x, y)[0, 1])
        if len(x) > 1 and np.std(x) > 0 and np.std(y) > 0
        else float("nan")
    )
    return {"slope": slope, "intercept": intercept, "r2": r2, "pearson": pearson}


def load_episodes() -> tuple[list[dict], list[tuple[str, str]]]:
    skipped = []
    episodes = []
    with open(CSV_PATH, newline="") as f:
        reader = csv.DictReader(f)
        print(f"CSV columns: {', '.join(reader.fieldnames or [])}")
        print(f"Trajectory dir: {TRAJ_DIR} (pattern: seed_*.json)\n")
        for row in reader:
            seed = row["seed"]
            ep = {
                "seed": int(seed),
                "obj_x": float(row["target_object_init_x"]),
                "obj_y": float(row["target_object_init_y"]),
                "obj_z": float(row["target_object_init_z"]),
                "success": parse_bool(row["success"]),
                "num_steps": int(row["num_steps"]),
                "traj_path": Path(row["ee_trajectory_path"]),
            }
            if not ep["traj_path"].is_file():
                alt = TRAJ_DIR / f"seed_{seed}.json"
                if alt.is_file():
                    ep["traj_path"] = alt
                else:
                    skipped.append((seed, "trajectory missing"))
                    continue
            episodes.append(ep)
    return episodes, skipped


def check_a(episodes: list[dict]) -> None:
    print("=" * 72)
    print("CHECK A — Failure vs displacement")
    print("=" * 72)

    cx = np.mean([e["obj_x"] for e in episodes])
    cy = np.mean([e["obj_y"] for e in episodes])
    print(f"Cluster center (mean object init): x={cx:.4f}, y={cy:.4f}\n")

    for e in episodes:
        e["dx"] = e["obj_x"] - cx
        e["dy"] = e["obj_y"] - cy
        e["abs_dx"] = abs(e["dx"])
        e["abs_dy"] = abs(e["dy"])
        e["displacement"] = float(np.hypot(e["dx"], e["dy"]))

    ranked = sorted(episodes, key=lambda e: e["displacement"], reverse=True)
    for rank, e in enumerate(ranked, start=1):
        e["disp_rank"] = rank

    print(
        f"{'rank':>4} {'seed':>4} {'obj_x':>8} {'obj_y':>8} "
        f"{'dx':>8} {'dy':>8} {'|dx|':>6} {'|dy|':>6} {'disp':>7} {'ok':>5} {'steps':>5}"
    )
    print("-" * 82)
    for e in ranked:
        print(
            f"{e['disp_rank']:4d} {e['seed']:4d} {e['obj_x']:8.4f} {e['obj_y']:8.4f} "
            f"{e['dx']:8.4f} {e['dy']:8.4f} {e['abs_dx']:6.4f} {e['abs_dy']:6.4f} "
            f"{e['displacement']:7.4f} {str(e['success']):>5} {e['num_steps']:5d}"
        )
    print()

    failures = [e for e in episodes if not e["success"]]
    print(f"Failures: {len(failures)} / {len(episodes)}")
    if failures:
        for e in failures:
            print(
                f"  seed {e['seed']}: displacement rank {e['disp_rank']}/{len(episodes)} "
                f"(disp={e['displacement']:.4f} m = {e['displacement']*100:.2f} cm, "
                f"|dx|={e['abs_dx']*100:.2f} cm, |dy|={e['abs_dy']*100:.2f} cm)"
            )
    print()

    # Correlation hints (not formal inference at n=20, n_fail=2)
    disp = np.array([e["displacement"] for e in episodes])
    steps = np.array([e["num_steps"] for e in episodes], dtype=float)
    success = np.array([1.0 if e["success"] else 0.0 for e in episodes])
    if len(episodes) > 2 and np.std(disp) > 0:
        r_steps = float(np.corrcoef(disp, steps)[0, 1])
        print(f"Pearson r(displacement, num_steps) over all episodes: {r_steps:.3f} (hint only)")
    print()

    print("Failure-rank answer:")
    if len(failures) == 0:
        print("  No failures in dataset.")
    elif len(failures) <= 2:
        ranks = ", ".join(f"seed {e['seed']} is rank {e['disp_rank']}" for e in failures)
        print(f"  {ranks} of {len(episodes)} by displacement.")
        top_k = max(3, len(failures))
        high_disp = [e for e in failures if e["disp_rank"] <= top_k]
        if len(high_disp) == len(failures):
            print(
                "  Both failures are in the top tier by displacement — a hint consistent "
                "with replay, but n=2 failures is too small for evidence."
            )
        elif len(high_disp) == 0:
            print(
                "  Failures are NOT among the highest-displacement episodes — "
                "does NOT support the simple replay story."
            )
        else:
            print(
                "  Mixed: some failures high-displacement, others not — "
                "does NOT cleanly support replay-from-displacement alone."
            )
    print()

    print("Plain-English read (CHECK A):")
    if len(failures) <= 2:
        print(
            "  Sample too small (2 failures) to be statistically meaningful — treat as a "
            "hint, not evidence."
        )
    if failures:
        max_fail_rank = max(e["disp_rank"] for e in failures)
        if max_fail_rank <= 3:
            print(
                "  Hint consistent with replay: failures occur at above-average displacement "
                f"(worst failure rank = {max_fail_rank}/{len(episodes)}), but confirm with "
                "more failures before claiming evidence."
            )
        elif max_fail_rank <= len(episodes) // 2:
            print(
                "  Ambiguous: failures are mid-to-high displacement but not exclusively "
                "the outliers — replay may contribute but is not the sole explanation."
            )
        else:
            print(
                "  Does NOT support the simple replay story; failures aren't explained by "
                "displacement alone."
            )
    print()


def check_b(episodes: list[dict]) -> None:
    print("=" * 72)
    print("CHECK B — Is y really ignored, or just a forgiving axis?")
    print("=" * 72)
    print(
        "Grasp point definition: EE [x,y,z] at global minimum-z timestep "
        "(earliest if tied) — same as analyze_grasp.py.\n"
    )

    skipped_grasp = []
    successes = []
    for e in episodes:
        if not e["success"]:
            continue
        try:
            traj = load_trajectory(e["traj_path"])
            gp = grasp_point(traj)
            e["grasp_x"] = float(gp[0])
            e["grasp_y"] = float(gp[1])
            e["grasp_z"] = float(gp[2])
            successes.append(e)
        except (ValueError, json.JSONDecodeError, OSError) as err:
            skipped_grasp.append((e["seed"], str(err)))

    if skipped_grasp:
        print("Skipped successes (grasp extraction failed):")
        for seed, reason in skipped_grasp:
            print(f"  seed={seed}: {reason}")
        print()

    n = len(successes)
    print(f"Successes used for grasp analysis: {n}\n")

    obj_x = np.array([e["obj_x"] for e in successes])
    obj_y = np.array([e["obj_y"] for e in successes])
    grasp_x = np.array([e["grasp_x"] for e in successes])
    grasp_y = np.array([e["grasp_y"] for e in successes])

    def range_cm(vals: np.ndarray) -> tuple[float, float, float]:
        return float(vals.min()), float(vals.max()), float(vals.max() - vals.min()) * 100

    ox0, ox1, ox_span = range_cm(obj_x)
    oy0, oy1, oy_span = range_cm(obj_y)
    gx0, gx1, gx_span = range_cm(grasp_x)
    gy0, gy1, gy_span = range_cm(grasp_y)

    print("Ranges (successful rollouts only):")
    print(f"  object_init_x: min={ox0:.4f}, max={ox1:.4f}, span={ox_span:.2f} cm")
    print(f"  object_init_y: min={oy0:.4f}, max={oy1:.4f}, span={oy_span:.2f} cm")
    print(f"  grasp_x:       min={gx0:.4f}, max={gx1:.4f}, span={gx_span:.2f} cm")
    print(f"  grasp_y:       min={gy0:.4f}, max={gy1:.4f}, span={gy_span:.2f} cm")
    print()

    stats_x = linear_stats(obj_x, grasp_x)
    stats_y = linear_stats(obj_y, grasp_y)
    print("Grasp ~ object_init regression (successes only):")
    print(
        f"  x: slope={stats_x['slope']:.4f}, intercept={stats_x['intercept']:.4f}, "
        f"R²={stats_x['r2']:.4f}, r={stats_x['pearson']:.4f}"
    )
    print(
        f"  y: slope={stats_y['slope']:.4f}, intercept={stats_y['intercept']:.4f}, "
        f"R²={stats_y['r2']:.4f}, r={stats_y['pearson']:.4f}"
    )
    print()

    ratio_x = gx_span / ox_span if ox_span > 0 else float("nan")
    ratio_y = gy_span / oy_span if oy_span > 0 else float("nan")
    print("Tracking ratio = (grasp range) / (object init range):")
    print(f"  x: {ratio_x:.3f}  (grasp spans {ratio_x*100:.0f}% of object-x variation)")
    print(f"  y: {ratio_y:.3f}  (grasp spans {ratio_y*100:.0f}% of object-y variation)")
    print()

    print("Per-episode paired table (successes):")
    print(f"{'seed':>4} {'obj_x':>8} {'obj_y':>8} {'grasp_x':>8} {'grasp_y':>8} {'steps':>5}")
    print("-" * 50)
    for e in sorted(successes, key=lambda x: x["seed"]):
        print(
            f"{e['seed']:4d} {e['obj_x']:8.4f} {e['obj_y']:8.4f} "
            f"{e['grasp_x']:8.4f} {e['grasp_y']:8.4f} {e['num_steps']:5d}"
        )
    print()

    # Displacement vs steps by axis (all episodes)
    print("Per-axis |displacement from center| vs num_steps (all episodes, hint only):")
    for axis in ("x", "y"):
        vals = np.array([e[f"abs_d{axis}"] for e in episodes]) * 100
        steps = np.array([e["num_steps"] for e in episodes], dtype=float)
        r = float(np.corrcoef(vals, steps)[0, 1]) if np.std(vals) > 0 else float("nan")
        print(f"  |d{axis}| vs steps: Pearson r = {r:.3f}")
    print()

    print("Interpretation (CHECK B — hints only, n=18 successes):")
    if ratio_y < 0.5 and oy_span < 5.0:
        print(
            f"  y-axis: grasp-y range ({gy_span:.1f} cm) is small relative to object-y "
            f"range ({oy_span:.1f} cm); ratio={ratio_y:.2f}. Grasp barely moves in y "
            "while object does — consistent with ignoring y, BUT object-y only varies "
            f"~{oy_span:.1f} cm which may be within grasp tolerance (confounded)."
        )
    elif ratio_y < 0.5:
        print(
            f"  y-axis: low tracking ratio ({ratio_y:.2f}) — grasp-y moves less than "
            "object-y; consistent with partial y-ignore, but check tolerance confound."
        )
    else:
        print(
            f"  y-axis: tracking ratio={ratio_y:.2f} — grasp does move in y somewhat; "
            "does not strongly support 'y completely ignored'."
        )

    if stats_x["slope"] > 0.4 and stats_x["r2"] > 0.3:
        print(
            f"  x-axis: slope={stats_x['slope']:.2f}, R²={stats_x['r2']:.2f}, ratio={ratio_x:.2f} "
            "— partial x tracking (under-tracks: slope < 1). Hints at grounding on x "
            "but not full 1:1 tracking; still not conclusive at n=18."
        )
    else:
        print(
            f"  x-axis: weak x tracking (slope={stats_x['slope']:.2f}, R²={stats_x['r2']:.2f}) "
            "— prior 'grounds on x' finding is NOT hardened; may be noise."
        )
    print()
    print(
        "CONFOUND FLAG: Whether ignoring y = cheating vs. y being a forgiving axis "
        "CANNOT be resolved from rollout data alone. It requires the LIBERO task "
        "success definition for libero_spatial task_id=0 — specifically the position "
        "tolerance on each axis for the place condition. This script flags the "
        "confound; it does not resolve it."
    )
    print()


def plot_displacement(episodes: list[dict]) -> None:
    disp_cm = np.array([e["displacement"] for e in episodes]) * 100
    steps = np.array([e["num_steps"] for e in episodes])
    colors = ["#2ca02c" if e["success"] else "#d62728" for e in episodes]

    fig, ax = plt.subplots(figsize=(7, 5))
    for e, c in zip(episodes, colors):
        ax.scatter(e["displacement"] * 100, e["num_steps"], c=c, s=70, zorder=3)
        ax.annotate(str(e["seed"]), (e["displacement"] * 100, e["num_steps"]), fontsize=7, xytext=(3, 3), textcoords="offset points")

    from matplotlib.lines import Line2D
    ax.legend(
        handles=[
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#2ca02c", markersize=8, label="success"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#d62728", markersize=8, label="failure"),
        ],
        loc="upper left",
    )
    ax.set_xlabel("Object init displacement from cluster center (cm)")
    ax.set_ylabel("num_steps")
    ax.set_title("Displacement vs rollout length")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_PLOT, dpi=150)
    print(f"Saved optional plot to {OUT_PLOT}")


def main() -> None:
    if not CSV_PATH.exists():
        sys.exit(f"Missing {CSV_PATH}")

    episodes, skipped = load_episodes()
    if skipped:
        print("Skipped episodes:")
        for seed, reason in skipped:
            print(f"  seed={seed}: {reason}")
        print()

    if len(episodes) == 0:
        sys.exit("No episodes loaded.")

    check_a(episodes)
    check_b(episodes)
    plot_displacement(episodes)


if __name__ == "__main__":
    main()
