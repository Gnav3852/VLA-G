#!/usr/bin/env python3
"""
Analysis 3: Distinguish "replay" from "out-of-distribution breakage".

Reads spike2_rollouts.csv + spike2_trajectories/rollout_*.json only.

Outputs:
  success_vs_distance.png
  traj_similarity_vs_object_distance.png
  stdout tables + combined verdict
"""

from __future__ import annotations

import csv
import json
import math
import sys
from itertools import combinations
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
CSV_PATH = ROOT / "spike2_rollouts.csv"
TRAJ_DIR = ROOT / "spike2_trajectories"
OUT_SUCCESS_PLOT = ROOT / "success_vs_distance.png"
OUT_TRAJ_PLOT = ROOT / "traj_similarity_vs_object_distance.png"


def configure(
    csv_path: Path,
    traj_dir: Path,
    out_success_plot: Path,
    out_traj_plot: Path,
) -> None:
    global CSV_PATH, TRAJ_DIR, OUT_SUCCESS_PLOT, OUT_TRAJ_PLOT
    CSV_PATH = csv_path
    TRAJ_DIR = traj_dir
    OUT_SUCCESS_PLOT = out_success_plot
    OUT_TRAJ_PLOT = out_traj_plot

# Commanded grid center from modal_perturbation_sweep.py (for reference).
COMMANDED_CENTER_X = -0.06
COMMANDED_CENTER_Y = 0.20

TRAJ_RESAMPLE_N = 100
TRAJ_METRIC = (
    f"mean per-step Euclidean distance after linear resampling to {TRAJ_RESAMPLE_N} points"
)


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


def resample_trajectory(traj: np.ndarray, n: int) -> np.ndarray:
    if len(traj) == 1:
        return np.repeat(traj, n, axis=0)
    t_old = np.linspace(0.0, 1.0, len(traj))
    t_new = np.linspace(0.0, 1.0, n)
    out = np.zeros((n, 3), dtype=np.float64)
    for dim in range(3):
        out[:, dim] = np.interp(t_new, t_old, traj[:, dim])
    return out


def traj_distance(a: np.ndarray, b: np.ndarray, n: int = TRAJ_RESAMPLE_N) -> float:
    ra = resample_trajectory(a, n)
    rb = resample_trajectory(b, n)
    return float(np.mean(np.linalg.norm(ra - rb, axis=1)))


def object_xy_distance(x1: float, y1: float, x2: float, y2: float) -> float:
    return float(math.hypot(x1 - x2, y1 - y2))


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def load_valid_rollouts() -> tuple[list[dict[str, Any]], list[tuple[str, str]], dict[str, int]]:
    rows: list[dict[str, Any]] = []
    skipped: list[tuple[str, str]] = []
    counts = {"commanded": 0, "valid": 0, "rejected": 0}

    with CSV_PATH.open(newline="") as f:
        reader = csv.DictReader(f)
        print(f"CSV columns: {', '.join(reader.fieldnames or [])}")
        print(f"Trajectory dir: {TRAJ_DIR} (pattern: rollout_*.json)\n")

        for raw in reader:
            counts["commanded"] += 1
            rollout_id = raw.get("rollout_id", "?")

            if raw.get("status", "ok") != "ok":
                counts["rejected"] += 1
                skipped.append((rollout_id, f"status={raw.get('status')} (excluded from valid set)"))
                continue

            counts["valid"] += 1
            traj_path = Path(raw["ee_trajectory_path"])
            if not traj_path.is_file():
                alt = TRAJ_DIR / f"rollout_{int(rollout_id):03d}.json"
                if alt.is_file():
                    traj_path = alt
                else:
                    counts["valid"] -= 1
                    skipped.append((rollout_id, f"trajectory missing: {traj_path}"))
                    continue

            try:
                traj = load_trajectory(traj_path)
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                counts["valid"] -= 1
                skipped.append((rollout_id, str(exc)))
                continue

            rows.append(
                {
                    "rollout_id": int(rollout_id),
                    "grid_x": float(raw["grid_x"]),
                    "grid_y": float(raw["grid_y"]),
                    "obj_x": float(raw["actual_object_init_x"]),
                    "obj_y": float(raw["actual_object_init_y"]),
                    "success": parse_bool(raw.get("success", "")),
                    "num_steps": int(raw["num_steps"]),
                    "traj": traj,
                }
            )

    rows.sort(key=lambda r: r["rollout_id"])
    return rows, skipped, counts


def choose_center(rows: list[dict[str, Any]]) -> tuple[float, float, str]:
    mean_x = float(np.mean([r["obj_x"] for r in rows]))
    mean_y = float(np.mean([r["obj_y"] for r in rows]))
    return mean_x, mean_y, "mean of actual_object_init (x,y) across valid rollouts"


def perturbation_distances(rows: list[dict[str, Any]], cx: float, cy: float) -> np.ndarray:
    return np.array([object_xy_distance(r["obj_x"], r["obj_y"], cx, cy) for r in rows])


def make_bins(distances_m: np.ndarray) -> list[tuple[str, float, float]]:
    dmax = float(distances_m.max())
    edges_cm = [0, 3, 6, 10, 15]
    if dmax * 100 > edges_cm[-1]:
        edges_cm.append(math.ceil(dmax * 100))
    bins: list[tuple[str, float, float]] = []
    for lo_cm, hi_cm in zip(edges_cm[:-1], edges_cm[1:]):
        label = f"{lo_cm}-{hi_cm}cm" if hi_cm < edges_cm[-1] else f"{hi_cm}cm+"
        if hi_cm == edges_cm[-1] and lo_cm == edges_cm[-2]:
            label = f"{lo_cm}cm+"
        bins.append((label, lo_cm / 100.0, hi_cm / 100.0))
    return bins


def bin_success_table(
    rows: list[dict[str, Any]], distances_m: np.ndarray
) -> list[dict[str, Any]]:
    table: list[dict[str, Any]] = []
    bins = make_bins(distances_m)
    for i, (label, lo, hi) in enumerate(bins):
        if i == len(bins) - 1:
            mask = distances_m >= lo
        else:
            mask = (distances_m >= lo) & (distances_m < hi)
        idx = np.where(mask)[0]
        n = len(idx)
        n_success = sum(1 for i in idx if rows[i]["success"])
        rate = n_success / n if n else float("nan")
        table.append(
            {
                "bin": label,
                "lo_m": lo,
                "hi_m": hi,
                "n_rollouts": n,
                "n_success": n_success,
                "success_rate": rate,
            }
        )
    return table


def plot_success_vs_distance(
    rows: list[dict[str, Any]],
    distances_m: np.ndarray,
    table: list[dict[str, Any]],
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    y = np.array([1.0 if r["success"] else 0.0 for r in rows])
    jitter = np.random.default_rng(0).uniform(-0.04, 0.04, size=len(y))
    ax.scatter(distances_m * 100, y + jitter, alpha=0.65, s=42, c="#4C72B0", edgecolors="none", label="rollouts")

    bin_centers = []
    bin_rates = []
    for row in table:
        if row["n_rollouts"] == 0:
            continue
        center = (row["lo_m"] + row["hi_m"]) / 2 * 100
        bin_centers.append(center)
        bin_rates.append(row["success_rate"])

    if bin_centers:
        ax.plot(bin_centers, bin_rates, "o-", color="#C44E52", linewidth=2, markersize=7, label="binned success rate")

    ax.set_xlabel("Perturbation distance from center (cm)")
    ax.set_ylabel("Success (0/1) / binned rate")
    ax.set_ylim(-0.15, 1.15)
    ax.set_title("Success vs perturbation distance (valid rollouts only)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_SUCCESS_PLOT, dpi=150)
    plt.close(fig)


def pairwise_trajectory_analysis(rows: list[dict[str, Any]]) -> dict[str, Any]:
    pairs: list[dict[str, Any]] = []
    for i, j in combinations(range(len(rows)), 2):
        ri, rj = rows[i], rows[j]
        obj_dist = object_xy_distance(ri["obj_x"], ri["obj_y"], rj["obj_x"], rj["obj_y"])
        td = traj_distance(ri["traj"], rj["traj"])
        si, sj = ri["success"], rj["success"]
        if si and sj:
            group = "both_succeeded"
        elif (not si) and (not sj):
            group = "both_failed"
        else:
            group = "mixed"
        pairs.append(
            {
                "i": ri["rollout_id"],
                "j": rj["rollout_id"],
                "object_distance_m": obj_dist,
                "traj_distance_m": td,
                "group": group,
            }
        )

    obj_d = np.array([p["object_distance_m"] for p in pairs])
    traj_d = np.array([p["traj_distance_m"] for p in pairs])

    by_group: dict[str, dict[str, float]] = {}
    for group in ("both_failed", "both_succeeded", "mixed"):
        mask = np.array([p["group"] == group for p in pairs])
        if mask.sum() < 2:
            by_group[group] = {"n_pairs": int(mask.sum()), "pearson": float("nan")}
        else:
            by_group[group] = {
                "n_pairs": int(mask.sum()),
                "pearson": pearson(obj_d[mask], traj_d[mask]),
                "mean_traj_distance_m": float(traj_d[mask].mean()),
            }

    return {
        "pairs": pairs,
        "object_distance_m": obj_d,
        "traj_distance_m": traj_d,
        "pearson": pearson(obj_d, traj_d),
        "mean_traj_distance_m": float(traj_d.mean()),
        "median_traj_distance_m": float(np.median(traj_d)),
        "by_group": by_group,
    }


def _cell_key(row: dict[str, Any]) -> tuple[float, float]:
    return (round(float(row["grid_x"]), 4), round(float(row["grid_y"]), 4))


def controlled_pairwise_analysis(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Split pairs to control for the robot/scene start (base_idx).

    - same_base_idx: same robot start, different object position -> OBJECT-DRIVEN.
      Correlating object_distance vs traj_distance here is the clean grounding test.
    - same_position: same grid cell, different base_idx -> NON-OBJECT BASELINE.
      Measures trajectory variance caused purely by robot-start changes (object held
      fixed), i.e. the noise floor the object-driven signal must beat.
    """
    same_base: list[dict[str, Any]] = []
    same_pos: list[dict[str, Any]] = []

    for i, j in combinations(range(len(rows)), 2):
        ri, rj = rows[i], rows[j]
        obj_dist = object_xy_distance(ri["obj_x"], ri["obj_y"], rj["obj_x"], rj["obj_y"])
        td = traj_distance(ri["traj"], rj["traj"])
        both_succ = bool(ri["success"]) and bool(rj["success"])
        rec = {
            "i": ri["rollout_id"],
            "j": rj["rollout_id"],
            "object_distance_m": obj_dist,
            "traj_distance_m": td,
            "both_succeeded": both_succ,
        }
        same_base_idx = ri.get("base_idx", -1) == rj.get("base_idx", -1) and ri.get("base_idx", -1) >= 0
        same_cell = _cell_key(ri) == _cell_key(rj)

        if same_cell and not same_base_idx:
            same_pos.append(rec)
        elif same_base_idx and not same_cell:
            same_base.append(rec)

    def summarize(pairs: list[dict[str, Any]]) -> dict[str, Any]:
        if not pairs:
            return {"n_pairs": 0, "pearson": float("nan"), "mean_traj_distance_m": float("nan")}
        obj_d = np.array([p["object_distance_m"] for p in pairs])
        traj_d = np.array([p["traj_distance_m"] for p in pairs])
        succ_pairs = [p for p in pairs if p["both_succeeded"]]
        out = {
            "n_pairs": len(pairs),
            "pearson": pearson(obj_d, traj_d),
            "mean_traj_distance_m": float(traj_d.mean()),
            "median_traj_distance_m": float(np.median(traj_d)),
            "n_success_pairs": len(succ_pairs),
        }
        if len(succ_pairs) >= 2:
            so = np.array([p["object_distance_m"] for p in succ_pairs])
            st = np.array([p["traj_distance_m"] for p in succ_pairs])
            out["success_pearson"] = pearson(so, st)
            out["success_mean_traj_distance_m"] = float(st.mean())
        else:
            out["success_pearson"] = float("nan")
            out["success_mean_traj_distance_m"] = float("nan")
        return out

    return {
        "same_base": same_base,
        "same_pos": same_pos,
        "same_base_summary": summarize(same_base),
        "same_pos_summary": summarize(same_pos),
    }


def plot_controlled_traj(controlled: dict[str, Any], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))

    sb = controlled["same_base"]
    if sb:
        xs = [p["object_distance_m"] * 100 for p in sb]
        ys = [p["traj_distance_m"] * 100 for p in sb]
        cols = ["#55A868" if p["both_succeeded"] else "#4C72B0" for p in sb]
        ax.scatter(xs, ys, alpha=0.5, s=30, c=cols, edgecolors="none",
                   label=f"same robot-start, diff object (n={len(sb)})")
        s = controlled["same_base_summary"]
        if not math.isnan(s["pearson"]) and len(sb) >= 2:
            xa = np.array(xs)
            slope, intercept = np.polyfit(xa, np.array(ys), 1)
            xr = np.linspace(xa.min(), xa.max(), 50)
            ax.plot(xr, slope * xr + intercept, "k-", linewidth=1.3,
                    label=f"OLS (r={s['pearson']:.2f})")

    sp = controlled["same_pos_summary"]
    if not math.isnan(sp["mean_traj_distance_m"]):
        ax.axhline(sp["mean_traj_distance_m"] * 100, color="#C44E52", linestyle="--", linewidth=1.5,
                   label=f"non-object baseline (same pos, diff start): {sp['mean_traj_distance_m']*100:.1f}cm")

    ax.set_xlabel("Object init distance between rollouts (cm)")
    ax.set_ylabel("Trajectory distance (cm)")
    ax.set_title("Base_idx-controlled: object-driven trajectory difference vs noise floor")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _centroid_path(trajs: list[np.ndarray], n: int = TRAJ_RESAMPLE_N) -> tuple[np.ndarray, np.ndarray]:
    """Return (centroid[n,3], per-timestep mean distance to centroid [n])."""
    stack = np.stack([resample_trajectory(t, n) for t in trajs], axis=0)  # [k, n, 3]
    centroid = stack.mean(axis=0)
    spread = np.linalg.norm(stack - centroid[None, :, :], axis=2).mean(axis=0)  # [n]
    return centroid, spread


def tube_width(trajs: list[np.ndarray], n: int = TRAJ_RESAMPLE_N) -> float:
    """Mean over time of the per-timestep spread to the centroid (the motion 'tube' width)."""
    if len(trajs) < 2:
        return float("nan")
    _, spread = _centroid_path(trajs, n)
    return float(spread.mean())


def plot_failed_overlay(rows: list[dict[str, Any]], out_path: Path, cx: float, cy: float) -> dict[str, Any]:
    """Overlay EE paths per base_idx (robot start held fixed), failed vs succeeded.

    Color encodes perturbation distance of the object from center; the object init is
    drawn as a star and the grasp point as an X in the same color. Replay-like failure =>
    grasp X's collapse together while object stars spread; grounding => grasp X's follow stars.
    Returns per-group tube widths for the FINDINGS writeup.
    """
    base_idxs = sorted({r.get("base_idx", -1) for r in rows if r.get("base_idx", -1) >= 0})
    if not base_idxs:
        base_idxs = [-1]

    dists = np.array(
        [object_xy_distance(r["obj_x"], r["obj_y"], cx, cy) for r in rows]
    )
    vmin, vmax = float(dists.min()) * 100, float(dists.max()) * 100
    cmap = plt.cm.viridis
    norm = plt.Normalize(vmin=vmin, vmax=vmax)

    ncols = len(base_idxs)
    fig, axes = plt.subplots(2, ncols, figsize=(4.2 * ncols, 8.4), squeeze=False)

    summary: dict[str, Any] = {"by_group": {}}

    for col, bidx in enumerate(base_idxs):
        for row_i, (want_success, gname) in enumerate([(False, "failed"), (True, "succeeded")]):
            ax = axes[row_i][col]
            sel = [
                r for r in rows
                if r.get("base_idx", -1) == bidx and bool(r["success"]) == want_success
            ]
            for r in sel:
                d_cm = object_xy_distance(r["obj_x"], r["obj_y"], cx, cy) * 100
                color = cmap(norm(d_cm))
                traj = r["traj"]
                ax.plot(traj[:, 0], traj[:, 1], "-", color=color, alpha=0.55, linewidth=1.0)
                ax.scatter([r["obj_x"]], [r["obj_y"]], marker="*", s=110, color=color,
                           edgecolors="k", linewidths=0.4, zorder=5)
                ax.scatter([r["grasp_x"]], [r["grasp_y"]], marker="x", s=45, color=color,
                           linewidths=1.6, zorder=6)

            trajs = [r["traj"] for r in sel]
            tw = tube_width(trajs) if len(trajs) >= 2 else float("nan")
            if len(trajs) >= 2:
                centroid, _ = _centroid_path(trajs)
                ax.plot(centroid[:, 0], centroid[:, 1], "-", color="k", linewidth=2.2,
                        alpha=0.9, zorder=7, label="centroid path")
                ax.legend(loc="upper right", fontsize=7)

            summary["by_group"][(gname, bidx)] = {"n": len(sel), "tube_width_m": tw}
            ax.set_title(f"{gname} | base_idx={bidx} (n={len(sel)}, tube={tw*100:.1f}cm)"
                         if not math.isnan(tw) else f"{gname} | base_idx={bidx} (n={len(sel)})",
                         fontsize=9)
            ax.set_xlabel("EE x (m)")
            ax.set_ylabel("EE y (m)")
            ax.grid(True, alpha=0.3)
            ax.set_aspect("equal", adjustable="datalim")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, shrink=0.6, location="right", pad=0.02)
    cbar.set_label("object perturbation distance from center (cm)")

    fig.suptitle(
        "Failed vs succeeded EE paths, robot start fixed per column\n"
        "star = object init, X = grasp point (same color = same object distance)",
        fontsize=11,
    )
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Aggregate tube widths pooled across base_idx (within-start, so start variance excluded).
    for gname, want_success in [("failed", False), ("succeeded", True)]:
        pooled = []
        for bidx in base_idxs:
            trajs = [
                r["traj"] for r in rows
                if r.get("base_idx", -1) == bidx and bool(r["success"]) == want_success
            ]
            if len(trajs) >= 2:
                pooled.append(tube_width(trajs))
        summary[gname + "_mean_tube_width_m"] = float(np.nanmean(pooled)) if pooled else float("nan")

    return summary


def plot_traj_vs_object(pairwise: dict[str, Any]) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {"both_failed": "#4C72B0", "both_succeeded": "#55A868", "mixed": "#C44E52"}
    for group in ("both_failed", "mixed", "both_succeeded"):
        pts = [p for p in pairwise["pairs"] if p["group"] == group]
        if not pts:
            continue
        xs = [p["object_distance_m"] * 100 for p in pts]
        ys = [p["traj_distance_m"] * 100 for p in pts]
        ax.scatter(xs, ys, alpha=0.45, s=28, c=colors[group], label=f"{group} (n={len(pts)})", edgecolors="none")

    ax.set_xlabel("Object init distance between rollouts (cm)")
    ax.set_ylabel("Trajectory distance (cm)\n(mean resampled per-step EE Euclidean)")
    ax.set_title("Trajectory distance vs object separation (all valid rollout pairs)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_TRAJ_PLOT, dpi=150)
    plt.close(fig)


def read_analysis1(distances_m: np.ndarray, table: list[dict[str, Any]], rows: list[dict[str, Any]]) -> str:
    n_success = sum(1 for r in rows if r["success"])
    success_dists = distances_m[[i for i, r in enumerate(rows) if r["success"]]]

    if n_success <= 2:
        caveat = (
            f"Only {n_success} successes — this analysis is severely underpowered; "
            "treat as a hint, not evidence."
        )
    else:
        caveat = ""

    far_bins = [b for b in table if b["lo_m"] >= 0.10]
    near_bins = [b for b in table if b["hi_m"] <= 0.06]
    far_success = sum(b["n_success"] for b in far_bins)
    near_success = sum(b["n_success"] for b in near_bins)
    far_n = sum(b["n_rollouts"] for b in far_bins)
    near_n = sum(b["n_rollouts"] for b in near_bins)

    if n_success == 0:
        body = "No successes at any distance — consistent with OOD breakage or a broken policy, but inconclusive alone."
    elif near_success > 0 and far_success == 0 and far_n > 0:
        body = (
            "Sharp cliff pattern: successes only near center, none in far bins → "
            "OOD breakage: the perturbation range likely exceeded the model's usable envelope. "
            "The low grasp-tracking is likely breakage, not replay."
        )
    elif n_success >= 2 and len(success_dists) >= 2 and float(success_dists.max() - success_dists.min()) > 0.03:
        body = (
            "Successes spread across perturbation distances → not a simple OOD cliff; replay remains plausible."
        )
    else:
        body = (
            "Success pattern is ambiguous: too few successes to infer a clean distance envelope. "
            "Cannot rule out OOD cliff or replay from success alone."
        )

    return f"{body} {caveat}".strip()


def read_analysis3(pairwise: dict[str, Any]) -> str:
    r = pairwise["pearson"]
    mean_td_cm = pairwise["mean_traj_distance_m"] * 100
    med_td_cm = pairwise["median_traj_distance_m"] * 100

    both_fail = pairwise["by_group"]["both_failed"]
    bf_mean = both_fail.get("mean_traj_distance_m", float("nan"))
    bf_mean_cm = bf_mean * 100 if not math.isnan(bf_mean) else float("nan")
    bf_r = both_fail.get("pearson", float("nan"))

    if mean_td_cm < 5.0 and abs(r) < 0.3:
        return (
            f"Trajectory distance stays low and flat (mean={mean_td_cm:.1f}cm, median={med_td_cm:.1f}cm, "
            f"r={r:.3f}) regardless of object separation → REPLAY signal: similar canned motion "
            f"even as the object moves."
        )
    if not math.isnan(bf_mean_cm) and bf_mean_cm < 6.0 and abs(bf_r) < 0.3:
        return (
            f"Among failed pairs only, trajectory distance stays low and flat (mean={bf_mean_cm:.1f}cm, "
            f"r={bf_r:.3f}) → REPLAY signal: canned motion even when the task fails."
        )
    if r > 0.5 and mean_td_cm > 8.0:
        return (
            f"Trajectory distance is high and rises with object separation (r={r:.3f}, mean={mean_td_cm:.1f}cm) "
            "→ GROUNDING signal: motion changes proportionally with object position."
        )
    if mean_td_cm > 8.0 and abs(r) < 0.3:
        return (
            f"Trajectory distance is moderately high (mean={mean_td_cm:.1f}cm) but weakly correlated with object "
            f"separation (r={r:.3f}) → leans OOD BREAKAGE (erratic motion), though failed-only mean "
            f"({bf_mean_cm:.1f}cm) is lower and muddies the read."
        )

    return (
        f"Mixed signal: mean traj distance={mean_td_cm:.1f}cm, r={r:.3f}, "
        f"failed-pair mean={bf_mean_cm:.1f}cm. "
        "Pattern does not cleanly match replay, grounding, or OOD breakage alone."
    )


def combined_verdict(
    rows: list[dict[str, Any]],
    distances_m: np.ndarray,
    table: list[dict[str, Any]],
    pairwise: dict[str, Any],
    usable_range_m: float,
) -> str:
    n_success = sum(1 for r in rows if r["success"])
    a1 = read_analysis1(distances_m, table, rows)
    a3 = read_analysis3(pairwise)
    r = pairwise["pearson"]
    mean_td_cm = pairwise["mean_traj_distance_m"] * 100
    both_fail = pairwise["by_group"]["both_failed"]
    bf_mean_cm = both_fail.get("mean_traj_distance_m", float("nan"))
    bf_mean_cm = bf_mean_cm * 100 if not math.isnan(bf_mean_cm) else float("nan")
    bf_r = both_fail.get("pearson", float("nan"))

    a1_ood = "OOD breakage" in a1 and "exceeded" in a1
    a3_replay = "REPLAY signal" in a3
    a3_ood = "OOD BREAKAGE" in a3
    a3_ground = "GROUNDING signal" in a3

    if a3_replay:
        label = "replay"
        detail = (
            f"Trajectories stay similar across object positions (failed-pair mean {bf_mean_cm:.1f}cm, "
            f"r={bf_r:.3f}) while grasp barely tracks the object. Likely a largely fixed motion."
        )
    elif a3_ground:
        label = "grounding"
        detail = "Trajectory differences scale with object separation; motion adapts to object position."
    elif a1_ood and a3_ood and (math.isnan(bf_mean_cm) or bf_mean_cm >= 6.0):
        label = "OOD breakage"
        detail = (
            f"Success collapses away from center ({n_success}/{len(rows)} succeeded; usable range "
            f"≈{usable_range_m*100:.1f}cm) and trajectories are moderately high-variance without tracking "
            f"object separation (mean {mean_td_cm:.1f}cm, r={r:.3f}). Low grasp-tracking likely reflects "
            "incoherent failure outside the training envelope, not memorized shortcuts."
        )
    else:
        label = "inconclusive"
        detail = (
            f"Analysis 1 hints at an OOD cliff ({n_success} successes, usable range ≈{usable_range_m*100:.0f}cm) "
            f"but is underpowered. Analysis 3 is ambiguous: overall mean traj distance {mean_td_cm:.1f}cm "
            f"(r={r:.3f}), yet failed-only pairs average {bf_mean_cm:.1f}cm with r={bf_r:.3f} — not a clean "
            "replay flat-line, not clear grounding, not obvious chaotic flailing. "
            f"Re-run with a tighter grid within ~{usable_range_m*100:.0f}cm of center and ≥50 rollouts."
        )

    return f"VERDICT: {label} — {detail}"


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Analysis 3: replay vs OOD breakage")
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
            ROOT / "spike3_success_vs_distance.png",
            ROOT / "spike3_traj_similarity_vs_object_distance.png",
        )

    rows, skipped, counts = load_valid_rollouts()

    print("=== Dataset ===")
    print(f"Commanded rollouts: {counts['commanded']}")
    print(f"Valid (status=ok):    {counts['valid']}")
    print(f"Rejected (status!=ok): {counts['rejected']}")
    print(f"Used for analysis:    {len(rows)} (valid rollouts with readable trajectories)")
    if skipped:
        print("\nSkipped rollouts:")
        for rid, reason in skipped:
            print(f"  rollout {rid}: {reason}")

    if len(rows) < 2:
        print("\nNeed at least 2 valid rollouts.", file=sys.stderr)
        return 1

    cx, cy, center_method = choose_center(rows)
    print(f"\nCenter for perturbation distance: {center_method}")
    print(f"  center = ({cx:.4f}, {cy:.4f}) m")
    print(
        f"  (commanded grid center for reference: ({COMMANDED_CENTER_X:.4f}, {COMMANDED_CENTER_Y:.4f}) m)"
    )

    distances_m = perturbation_distances(rows, cx, cy)
    n_success = sum(1 for r in rows if r["success"])

    print("\n=== ANALYSIS 1 — Success rate vs perturbation distance ===")
    print(f"Valid rollouts: {len(rows)}, successes: {n_success}\n")
    print(f"{'bin':<12} {'n_rollouts':>10} {'n_success':>10} {'success_rate':>14}")
    print("-" * 48)
    table = bin_success_table(rows, distances_m)
    for row in table:
        rate_str = f"{row['success_rate']:.3f}" if row["n_rollouts"] else "n/a"
        print(
            f"{row['bin']:<12} {row['n_rollouts']:>10} {row['n_success']:>10} {rate_str:>14}"
        )

    success_mask = np.array([r["success"] for r in rows])
    if success_mask.any():
        usable_range_m = float(distances_m[success_mask].max())
    else:
        usable_range_m = 0.0
    print(f"\nUsable range (max perturbation distance with any success): {usable_range_m*100:.2f} cm")

    print("\nPer-rollout perturbation distances:")
    print(f"{'id':>4} {'dist_cm':>8} {'success':>8} {'obj_x':>10} {'obj_y':>10}")
    for r, d in zip(rows, distances_m):
        print(
            f"{r['rollout_id']:>4} {d*100:>8.2f} {str(r['success']):>8} "
            f"{r['obj_x']:>10.4f} {r['obj_y']:>10.4f}"
        )

    a1_read = read_analysis1(distances_m, table, rows)
    print(f"\nAnalysis 1 read: {a1_read}")

    plot_success_vs_distance(rows, distances_m, table)
    print(f"\nWrote {OUT_SUCCESS_PLOT.name}")

    print("\n=== ANALYSIS 3 — Trajectory similarity across object positions ===")
    print(f"Trajectory metric: {TRAJ_METRIC}")
    print(f"Pairs: C({len(rows)},2) = {len(rows)*(len(rows)-1)//2}\n")

    pairwise = pairwise_trajectory_analysis(rows)
    print(f"Overall pairwise correlation (object_distance vs traj_distance): r = {pairwise['pearson']:.4f}")
    print(f"Mean trajectory distance:   {pairwise['mean_traj_distance_m']*100:.2f} cm")
    print(f"Median trajectory distance: {pairwise['median_traj_distance_m']*100:.2f} cm")

    print("\nPer-group correlations (bonus):")
    for group in ("both_failed", "both_succeeded", "mixed"):
        g = pairwise["by_group"][group]
        r_str = f"{g['pearson']:.4f}" if not math.isnan(g["pearson"]) else "n/a (too few pairs)"
        extra = ""
        if "mean_traj_distance_m" in g:
            extra = f", mean traj dist = {g['mean_traj_distance_m']*100:.2f} cm"
        print(f"  {group:<16} n_pairs={g['n_pairs']:<4} r={r_str}{extra}")

    a3_read = read_analysis3(pairwise)
    print(f"\nAnalysis 3 read: {a3_read}")

    plot_traj_vs_object(pairwise)
    print(f"Wrote {OUT_TRAJ_PLOT.name}")

    print("\n=== Combined verdict ===")
    print(combined_verdict(rows, distances_m, table, pairwise, usable_range_m))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
