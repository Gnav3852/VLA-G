"""LIBERO spatial task presets for perturbation sweeps."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class TaskPreset:
    task_id: int
    instruction: str
    target_object: str
    center_x: float
    center_y: float
    openvla_csv: Path
    openvla_traj_dir: Path
    pi05_csv: Path
    pi05_traj_dir: Path
    controls_dir: Path
    comparison_md: Path


TASK_PRESETS: dict[int, TaskPreset] = {
    0: TaskPreset(
        task_id=0,
        instruction=(
            "Pick the akita black bowl between the plate and the ramekin and place it on the plate"
        ),
        target_object="akita_black_bowl_1",
        center_x=-0.06,
        center_y=0.20,
        openvla_csv=ROOT / "spike3_rollouts.csv",
        openvla_traj_dir=ROOT / "spike3_trajectories",
        pi05_csv=ROOT / "pi05_rollouts.csv",
        pi05_traj_dir=ROOT / "pi05_trajectories",
        controls_dir=ROOT / "controls",
        comparison_md=ROOT / "COMPARISON.md",
    ),
    2: TaskPreset(
        task_id=2,
        instruction="Pick the akita black bowl from table center and place it on the plate",
        target_object="akita_black_bowl_1",
        center_x=-0.075,
        center_y=0.0,
        openvla_csv=ROOT / "openvla_task2_rollouts.csv",
        openvla_traj_dir=ROOT / "openvla_task2_trajectories",
        pi05_csv=ROOT / "pi05_task2_rollouts.csv",
        pi05_traj_dir=ROOT / "pi05_task2_trajectories",
        controls_dir=ROOT / "controls_task2",
        comparison_md=ROOT / "COMPARISON_task2.md",
    ),
}


def get_task_preset(task_id: int) -> TaskPreset:
    if task_id not in TASK_PRESETS:
        known = ", ".join(str(k) for k in sorted(TASK_PRESETS))
        raise ValueError(f"Unknown task_id={task_id}; known: {known}")
    return TASK_PRESETS[task_id]


# Wide-range position sweep outputs (FINAL RUN)
WIDE_HALF_WIDTHS_M: tuple[float, ...] = (0.10, 0.15, 0.20)


def wide_sweep_paths(task_id: int, model: str, half_width_m: float) -> tuple[Path, Path]:
    """CSV + trajectory dir for a wide-range sweep at ±half_width_m."""
    if model not in ("openvla", "pi05"):
        raise ValueError(f"model must be openvla or pi05, got {model!r}")
    half_cm = int(round(half_width_m * 100))
    tag = f"wide_task{task_id}_{model}_w{half_cm}cm"
    return ROOT / f"{tag}_rollouts.csv", ROOT / f"{tag}_trajectories"
