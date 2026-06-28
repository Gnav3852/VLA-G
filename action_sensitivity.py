"""
Action-distribution sensitivity metrics from rollout trajectories.

We proxy per-step actions as EE position deltas (velocity) because historical
rollouts store ee_trajectory only. Sensitivity = mean L2 shift in velocity
between paired rollouts that share robot start (base_idx) but differ in object
position, normalized by object displacement.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

import analyze_check3 as check3

RESAMPLE_N = 100


def load_trajectory(path: Path) -> np.ndarray:
    with path.open() as f:
        arr = np.asarray(json.load(f), dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"expected (T,3) trajectory, got {arr.shape}")
    return arr


def trajectory_velocity(traj: np.ndarray) -> np.ndarray:
    if len(traj) < 2:
        return np.zeros((0, 3), dtype=np.float64)
    return np.diff(traj, axis=0)


def resample_velocity(traj: np.ndarray, n: int = RESAMPLE_N) -> np.ndarray:
    vel = trajectory_velocity(traj)
    if len(vel) == 0:
        return vel
    resampled_traj = check3.resample_trajectory(traj, n + 1)
    return trajectory_velocity(resampled_traj)


def mean_velocity_l2_shift(traj_a: np.ndarray, traj_b: np.ndarray, n: int = RESAMPLE_N) -> float:
    va = resample_velocity(traj_a, n)
    vb = resample_velocity(traj_b, n)
    if len(va) == 0 or len(vb) == 0:
        return float("nan")
    m = min(len(va), len(vb))
    return float(np.mean(np.linalg.norm(va[:m] - vb[:m], axis=1)))


def object_xy_distance(row_a: dict[str, Any], row_b: dict[str, Any]) -> float:
    return float(math.hypot(row_a["obj_x"] - row_b["obj_x"], row_a["obj_y"] - row_b["obj_y"]))


def attach_trajectories(rows: list[dict[str, Any]], traj_dir: Path) -> None:
    for row in rows:
        path = traj_dir / f"rollout_{row['rollout_id']:03d}.json"
        row["traj"] = load_trajectory(path)


def reference_row_for_base(rows: list[dict[str, Any]], base_idx: int, center_x: float, center_y: float) -> dict[str, Any] | None:
    candidates = [r for r in rows if r.get("base_idx") == base_idx]
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda r: math.hypot(float(r["obj_x"]) - center_x, float(r["obj_y"]) - center_y),
    )


def pairwise_sensitivity_rows(
    rows: list[dict[str, Any]],
    center_x: float,
    center_y: float,
) -> list[dict[str, Any]]:
    """Same-base_idx pairs: sensitivity = velocity L2 shift / object displacement."""
    out: list[dict[str, Any]] = []
    base_idxs = sorted({int(r["base_idx"]) for r in rows})
    for base_idx in base_idxs:
        ref = reference_row_for_base(rows, base_idx, center_x, center_y)
        if ref is None or "traj" not in ref:
            continue
        for row in rows:
            if row["base_idx"] != base_idx or row["rollout_id"] == ref["rollout_id"]:
                continue
            obj_dist = object_xy_distance(ref, row)
            if obj_dist <= 1e-6:
                continue
            shift = mean_velocity_l2_shift(ref["traj"], row["traj"])
            if math.isnan(shift):
                continue
            out.append(
                {
                    "base_idx": base_idx,
                    "ref_rollout_id": ref["rollout_id"],
                    "rollout_id": row["rollout_id"],
                    "object_distance_m": obj_dist,
                    "velocity_l2_shift_m": shift,
                    "sensitivity_per_m": shift / obj_dist,
                    "ref_success": bool(ref["success"]),
                    "success": bool(row["success"]),
                }
            )
    return out


def raw_sensitivity_score(pairs: list[dict[str, Any]]) -> float:
    vals = [p["sensitivity_per_m"] for p in pairs if not math.isnan(p["sensitivity_per_m"])]
    return float(np.mean(vals)) if vals else float("nan")


def calibrate_score(raw: float, anchor_low: float, anchor_high: float) -> float:
    if math.isnan(raw) or math.isnan(anchor_low) or math.isnan(anchor_high):
        return float("nan")
    if anchor_high <= anchor_low:
        return float("nan")
    return float(np.clip((raw - anchor_low) / (anchor_high - anchor_low), 0.0, 1.0))


def position_sensitivity_summary(
    rows: list[dict[str, Any]],
    center_x: float,
    center_y: float,
    *,
    anchor_low: float | None = None,
    anchor_high: float | None = None,
) -> dict[str, Any]:
    pairs = pairwise_sensitivity_rows(rows, center_x, center_y)
    raw = raw_sensitivity_score(pairs)
    succ_pairs = [p for p in pairs if p["success"] and p["ref_success"]]
    raw_succ = raw_sensitivity_score(succ_pairs)

    calibrated = float("nan")
    calibrated_succ = float("nan")
    if anchor_low is not None and anchor_high is not None:
        calibrated = calibrate_score(raw, anchor_low, anchor_high)
        calibrated_succ = calibrate_score(raw_succ, anchor_low, anchor_high)

    return {
        "n_pairs": len(pairs),
        "n_success_pairs": len(succ_pairs),
        "raw_sensitivity_per_m": raw,
        "raw_sensitivity_success_only": raw_succ,
        "calibrated_0_1": calibrated,
        "calibrated_success_only_0_1": calibrated_succ,
        "pairs": pairs,
    }


def calibration_anchors_from_controls(controls_dir: Path) -> tuple[float, float]:
    """Use min/max raw sensitivity across the gain sweep as anchors."""
    import analyze_perturbation as perturb

    manifest = json.loads((controls_dir / "manifest.json").read_text())
    raw_vals: list[float] = []

    for entry in manifest:
        ctrl_dir = Path(entry["path"])
        csv_path = ctrl_dir / "rollouts.csv"
        traj_dir = ctrl_dir / "trajectories"
        if not csv_path.is_file():
            continue
        perturb.configure(csv_path, traj_dir, controls_dir / "_noop.png", label=entry["name"])
        rows, _ = perturb.read_rows()
        attach_trajectories(rows, traj_dir)
        cx = float(np.mean([r["obj_x"] for r in rows]))
        cy = float(np.mean([r["obj_y"] for r in rows]))
        summary = position_sensitivity_summary(rows, cx, cy)
        raw = summary["raw_sensitivity_per_m"]
        if not math.isnan(raw):
            raw_vals.append(raw)

    if len(raw_vals) < 2:
        return float("nan"), float("nan")
    return float(min(raw_vals)), float(max(raw_vals))
