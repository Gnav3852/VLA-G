#!/usr/bin/env python3
"""
Generate synthetic detector validation controls from a recorded successful trajectory.

Positive control (g=0): replay T0 verbatim; object varies => grasp slope ~0.
Oracle (g=1): grasp tracks object 1:1 via ramped xy shift => grasp slope ~1.
Intermediate g in {0.25, 0.5, 0.75} for calibration sweep.

Run:
  ./.venv/bin/python make_controls.py
  ./.venv/bin/python make_controls.py --task-id 2 --source-rollout-id 0
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from libero_task_config import TaskPreset, get_task_preset

ROOT = Path(__file__).resolve().parent

# Recorded successful motion near grid center (rollout 6 on task 0).
DEFAULT_SOURCE_ROLLOUT_ID = 6

GAINS = [0.0, 0.25, 0.5, 0.75, 1.0]
NOISE_LEVELS = ("clean", "noisy")
NOISE_SIGMA_M = 0.005  # 0.5 cm per-step xy noise
SUCCESS_RADIUS_M = 0.03  # 3 cm grasp tolerance

CSV_FIELDNAMES = [
    "rollout_id",
    "grid_x",
    "grid_y",
    "base_idx",
    "actual_object_init_x",
    "actual_object_init_y",
    "actual_object_init_z",
    "override_error_m",
    "grasp_point_x",
    "grasp_point_y",
    "grasp_point_z",
    "success",
    "num_steps",
    "ee_trajectory_path",
    "status",
    "error",
]


def load_trajectory(path: Path) -> np.ndarray:
    with path.open() as f:
        return np.asarray(json.load(f), dtype=np.float64)


def grasp_index(traj: np.ndarray) -> int:
    z = traj[:, 2]
    return int(np.where(z == z.min())[0][0])


def load_spike3_grid(csv_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with csv_path.open(newline="") as f:
        for raw in csv.DictReader(f):
            if raw.get("status", "ok") != "ok":
                continue
            rows.append(
                {
                    "rollout_id": int(raw["rollout_id"]),
                    "grid_x": float(raw["grid_x"]),
                    "grid_y": float(raw["grid_y"]),
                    "base_idx": int(raw["base_idx"]),
                    "obj_x": float(raw["actual_object_init_x"]),
                    "obj_y": float(raw["actual_object_init_y"]),
                    "obj_z": float(raw["actual_object_init_z"]),
                }
            )
    rows.sort(key=lambda r: r["rollout_id"])
    return rows


def load_source_motion(
    csv_path: Path, traj_dir: Path, source_rollout_id: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, dict[str, float]]:
    traj_path = traj_dir / f"rollout_{source_rollout_id:03d}.json"
    traj = load_trajectory(traj_path)
    k = grasp_index(traj)
    gp = traj[k].copy()

    with csv_path.open(newline="") as f:
        for raw in csv.DictReader(f):
            if int(raw["rollout_id"]) == source_rollout_id:
                o0 = np.array(
                    [
                        float(raw["actual_object_init_x"]),
                        float(raw["actual_object_init_y"]),
                    ],
                    dtype=np.float64,
                )
                meta = {
                    "grid_x": float(raw["grid_x"]),
                    "grid_y": float(raw["grid_y"]),
                    "obj_z": float(raw["actual_object_init_z"]),
                }
                break
        else:
            raise RuntimeError(f"source rollout {source_rollout_id} not in csv")

    return traj, o0, gp, k, meta


def build_trajectory(
    t0: np.ndarray,
    grasp_k: int,
    delta_xy: np.ndarray,
    sigma: float,
    rng: np.random.Generator,
) -> np.ndarray:
    n = len(t0)
    out = t0.copy()
    ramp = np.zeros(n, dtype=np.float64)
    if grasp_k > 0:
        ramp[: grasp_k + 1] = np.linspace(0.0, 1.0, grasp_k + 1)
    else:
        ramp[:] = 1.0
    ramp[grasp_k:] = 1.0
    out[:, 0] += ramp * delta_xy[0]
    out[:, 1] += ramp * delta_xy[1]
    if sigma > 0:
        noise = rng.normal(0.0, sigma, size=(n, 2))
        out[:, 0] += noise[:, 0]
        out[:, 1] += noise[:, 1]
    return out


def dataset_name(g: float, noise_label: str) -> str:
    g_str = f"{g:.2f}".replace(".", "p")
    return f"ctrl_g{g_str}_{noise_label}"


def write_dataset(
    grid: list[dict[str, Any]],
    t0: np.ndarray,
    o0: np.ndarray,
    gp0: np.ndarray,
    grasp_k: int,
    obj_z_default: float,
    gain: float,
    noise_label: str,
    sigma: float,
    seed: int,
    controls_dir: Path,
) -> Path:
    name = dataset_name(gain, noise_label)
    out_dir = controls_dir / name
    traj_dir = out_dir / "trajectories"
    traj_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "rollouts.csv"

    rng = np.random.default_rng(seed)
    rows_out: list[dict[str, Any]] = []

    for spec in grid:
        obj = np.array([spec["obj_x"], spec["obj_y"]], dtype=np.float64)
        delta = gain * (obj - o0)
        traj = build_trajectory(t0, grasp_k, delta, sigma, rng)
        k = grasp_index(traj)
        gp = traj[k]
        dist = float(np.linalg.norm(gp[:2] - obj))
        success = dist < SUCCESS_RADIUS_M

        rid = spec["rollout_id"]
        traj_path = traj_dir / f"rollout_{rid:03d}.json"
        traj_path.write_text(json.dumps(traj.tolist()))

        rows_out.append(
            {
                "rollout_id": rid,
                "grid_x": spec["grid_x"],
                "grid_y": spec["grid_y"],
                "base_idx": spec["base_idx"],
                "actual_object_init_x": spec["obj_x"],
                "actual_object_init_y": spec["obj_y"],
                "actual_object_init_z": spec.get("obj_z", obj_z_default),
                "override_error_m": 0.0,
                "grasp_point_x": float(gp[0]),
                "grasp_point_y": float(gp[1]),
                "grasp_point_z": float(gp[2]),
                "success": success,
                "num_steps": len(traj),
                "ee_trajectory_path": str(traj_path),
                "status": "ok",
                "error": "",
            }
        )

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows_out)

    n_succ = sum(1 for r in rows_out if r["success"])
    print(
        f"Wrote {name}: n={len(rows_out)}, successes={n_succ}, "
        f"gain={gain}, sigma={sigma*100:.1f}cm"
    )
    return out_dir


def find_successful_rollout(csv_path: Path, traj_dir: Path) -> int:
    """Pick a successful rollout near grid center for canned motion T0."""
    with csv_path.open(newline="") as f:
        rows = [r for r in csv.DictReader(f) if r.get("status", "ok") == "ok" and r.get("success") == "True"]
    if not rows:
        raise SystemExit(f"No successful rollouts in {csv_path}")
    rows.sort(key=lambda r: abs(float(r["grid_x"])) + abs(float(r["grid_y"])))
    rid = int(rows[0]["rollout_id"])
    traj_path = traj_dir / f"rollout_{rid:03d}.json"
    if not traj_path.is_file():
        raise SystemExit(f"Missing trajectory for rollout {rid}: {traj_path}")
    return rid


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate detector validation controls")
    parser.add_argument("--task-id", type=int, default=0, help="LIBERO spatial task id")
    parser.add_argument(
        "--source-rollout-id",
        type=int,
        default=None,
        help="Successful rollout to use as canned motion T0 (auto-detect if omitted)",
    )
    args = parser.parse_args()
    preset: TaskPreset = get_task_preset(args.task_id)
    csv_path = preset.openvla_csv
    traj_dir = preset.openvla_traj_dir
    controls_dir = preset.controls_dir

    if not csv_path.is_file():
        raise SystemExit(f"Missing {csv_path}")

    source_rollout_id = args.source_rollout_id
    if source_rollout_id is None:
        if args.task_id == 0:
            source_rollout_id = DEFAULT_SOURCE_ROLLOUT_ID
        else:
            source_rollout_id = find_successful_rollout(csv_path, traj_dir)
            print(f"Auto-selected source rollout {source_rollout_id}")

    grid = load_spike3_grid(csv_path)
    t0, o0, gp0, grasp_k, meta = load_source_motion(csv_path, traj_dir, source_rollout_id)
    print(f"Source: rollout {source_rollout_id}, grasp_k={grasp_k}, o0=({o0[0]:.4f}, {o0[1]:.4f})")
    print(f"Grid positions: {len(grid)}")

    controls_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, Any]] = []

    for g in GAINS:
        for noise_label in NOISE_LEVELS:
            sigma = 0.0 if noise_label == "clean" else NOISE_SIGMA_M
            seed = int(g * 1000) + (0 if noise_label == "clean" else 1)
            out_dir = write_dataset(
                grid=grid,
                t0=t0,
                o0=o0,
                gp0=gp0,
                grasp_k=grasp_k,
                obj_z_default=meta["obj_z"],
                gain=g,
                noise_label=noise_label,
                sigma=sigma,
                seed=seed,
                controls_dir=controls_dir,
            )
            manifest.append(
                {
                    "name": dataset_name(g, noise_label),
                    "gain": g,
                    "noise": noise_label,
                    "path": str(out_dir),
                }
            )

    (controls_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nManifest: {controls_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
