"""
Build PROXY PatchMask behavioral scores from position sensitivity × VLA-Trace ΔSR.

WARNING: CIRCULAR — see CIRCULARITY_AUDIT.md. Do not use for validation; live rollouts only.
"""

from __future__ import annotations

import json
from pathlib import Path

import vla_trace_ground_truth as vtg
from per_condition_correlation import SETTING_TO_VARIANT

ROOT = Path(__file__).resolve().parent


def build_local_proxy_scores(position_json: Path) -> list[dict]:
    pos = json.loads(position_json.read_text())
    model_cals = {
        "openvla": pos["models"]["openvla"]["calibrated_0_1"],
        "pi05": pos["models"]["pi05"]["calibrated_0_1"],
        "openvla_oft": 0.85,
        "x_vla": 0.90,
    }
    # Fall back to raw if calibration saturated
    for key in ("openvla", "pi05"):
        cal = model_cals[key]
        raw = pos["models"][key]["raw_sensitivity_per_m"]
        if cal >= 0.999 or cal <= 0.001:
            model_cals[key] = raw

    wb = vtg.patchmask_conditions()
    spatial = [r for r in wb if r["suite"] == "libero_spatial"]
    max_delta_by_model: dict[str, float] = {}
    for row in spatial:
        max_delta_by_model[row["model"]] = max(
            max_delta_by_model.get(row["model"], 0.0), row["delta_sr"]
        )

    conditions = []
    for row in spatial:
        model = row["model"]
        if model not in model_cals:
            continue
        variant, mode = SETTING_TO_VARIANT.get(row["setting"], (row["setting"], "unknown"))
        denom = max_delta_by_model.get(model, 1.0) or 1.0
        rel = row["delta_sr"] / denom
        score = float(min(1.0, max(0.0, model_cals[model] * rel)))
        conditions.append(
            {
                "model": model,
                "variant": variant,
                "mode": mode,
                "suite": row["suite"],
                "setting": row["setting"],
                "action_sensitivity": score,
                "raw_sensitivity": score,
                "source": "position_scaled_proxy",
            }
        )
    return conditions
