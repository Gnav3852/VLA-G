"""
Per-condition correlation frame: align VLA-Trace PatchMask ΔSR with our behavioral
action-sensitivity under matching (model, mask_type, suite) conditions.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

import vla_trace_ground_truth as vtg

ROOT = Path(__file__).resolve().parent

# Map VLA-Trace table4 setting keys to PatchMask variant + mode.
SETTING_TO_VARIANT = {
    "target_bg": ("mask_target", "background_fill"),
    "target_black": ("mask_target", "black"),
    "target_mosaic": ("mask_target", "mosaic"),
    "gripper_bg": ("mask_gripper", "background_fill"),
    "gripper_black": ("mask_gripper", "black"),
    "gripper_mosaic": ("mask_gripper", "mosaic"),
    "robot_bg": ("mask_robot", "background_fill"),
    "robot_black": ("mask_robot", "black"),
    "robot_mosaic": ("mask_robot", "mosaic"),
    "robot_wo_gripper_bg": ("mask_robot_exc_gripper", "background_fill"),
    "robot_wo_gripper_black": ("mask_robot_exc_gripper", "black"),
    "robot_wo_gripper_mosaic": ("mask_robot_exc_gripper", "mosaic"),
    "background_black": ("mask_background", "black"),
    "background_mosaic": ("mask_background", "mosaic"),
}


@dataclass(frozen=True)
class ConditionKey:
    model: str
    variant: str
    mode: str
    suite: str


def load_behavioral_scores(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    data = json.loads(path.read_text())
    return data.get("conditions", [])


def white_box_conditions() -> list[dict[str, Any]]:
    rows = []
    for row in vtg.patchmask_conditions():
        variant, mode = SETTING_TO_VARIANT.get(row["setting"], (row["setting"], "unknown"))
        rows.append(
            {
                **row,
                "variant": variant,
                "mode": mode,
                "condition_key": f"{row['model']}|{variant}|{mode}|{row['suite']}",
            }
        )
    return rows


def align_conditions(
    white_box: list[dict[str, Any]],
    behavioral: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    beh_map = {
        (b["model"], b["variant"], b["mode"], b["suite"]): b for b in behavioral
    }
    aligned = []
    for wb in white_box:
        key = (wb["model"], wb["variant"], wb["mode"], wb["suite"])
        beh = beh_map.get(key)
        if beh is None:
            continue
        aligned.append(
            {
                **wb,
                "action_sensitivity": beh.get(
                    "action_sensitivity",
                    beh.get("raw_l2_shift", beh.get("calibrated_0_1")),
                ),
                "action_sensitivity_raw": beh.get(
                    "raw_sensitivity",
                    beh.get("raw_l2_shift", beh.get("raw_sensitivity_per_m")),
                ),
                "raw_l2_shift": beh.get("raw_l2_shift"),
                "normalized_l2_shift": beh.get("normalized_l2_shift"),
                "n_rollouts": beh.get("n_rollouts"),
                "behavioral_source": beh.get("source"),
            }
        )
    return aligned


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3:
        return float("nan")
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    if np.std(rx) == 0 or np.std(ry) == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def correlation_report(aligned: list[dict[str, Any]]) -> dict[str, Any]:
    if not aligned:
        return {"n": 0, "pearson_r": float("nan"), "spearman_rho": float("nan")}

    x = np.asarray([r["delta_sr"] for r in aligned], dtype=np.float64)
    y = np.asarray([r["action_sensitivity"] for r in aligned], dtype=np.float64)
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]

    r = pearson(x, y)
    rho = spearman(x, y)

    # Partial correlation controlling for baseline SR
    baseline = np.asarray([aligned[i]["baseline_sr"] for i in range(len(aligned)) if mask[i]], dtype=np.float64)
    if len(x) >= 4 and np.std(baseline) > 0:
        x_res = x - np.polyval(np.polyfit(baseline, x, 1), baseline)
        y_res = y - np.polyval(np.polyfit(baseline, y, 1), baseline)
        partial_r = pearson(x_res, y_res)
    else:
        partial_r = float("nan")

    return {
        "n": int(len(x)),
        "pearson_r": r,
        "r_squared": r * r if not math.isnan(r) else float("nan"),
        "spearman_rho": rho,
        "partial_r_controlling_baseline": partial_r,
    }


def per_model_rank_correlation(
    position_scores: dict[str, float],
    white_box_metric: str = "patchmask_target_bg_avg_drop",
) -> dict[str, Any]:
    derived = {r["model"]: r for r in vtg.derived_per_model()}
    models = sorted(set(position_scores) & set(derived))
    if len(models) < 2:
        return {"n_models": len(models), "spearman_rho": float("nan")}

    y = np.asarray([position_scores[m] for m in models], dtype=np.float64)
    x = np.asarray([derived[m].get(white_box_metric, float("nan")) for m in models], dtype=np.float64)
    return {"n_models": len(models), "models": models, "spearman_rho": spearman(x, y)}
