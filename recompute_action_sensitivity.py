#!/usr/bin/env python3
"""
Phase 1: recompute pi0.5 / OpenVLA position action-sensitivity on existing sweeps.

Compares action-sensitivity separation vs VLA-Trace white-box scalars that barely
separate the two models (visual_dep 1.00 vs 0.98, attention mass 0.63 vs 0.59).

Run:
  ./.venv/bin/python recompute_action_sensitivity.py
  ./.venv/bin/python recompute_action_sensitivity.py --task-id 2
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import analyze_perturbation as perturb
import numpy as np

import action_sensitivity as sens
import vla_trace_ground_truth as vtg
from libero_task_config import get_task_preset

ROOT = Path(__file__).resolve().parent


def load_model_rows(csv_path: Path, traj_dir: Path, label: str) -> list[dict[str, Any]]:
    perturb.configure(csv_path, traj_dir, ROOT / "_noop_grasp.png", label=label)
    rows, _ = perturb.read_rows()
    sens.attach_trajectories(rows, traj_dir)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Recompute action-distribution sensitivity")
    parser.add_argument("--task-id", type=int, default=0)
    args = parser.parse_args()
    preset = get_task_preset(args.task_id)

    anchor_low, anchor_high = sens.calibration_anchors_from_controls(preset.controls_dir)
    derived = {r["model"]: r for r in vtg.derived_per_model() if r["model"] in ("pi05", "openvla")}

    datasets = {
        "openvla": (preset.openvla_csv, preset.openvla_traj_dir),
        "pi05": (preset.pi05_csv, preset.pi05_traj_dir),
    }

    results: dict[str, Any] = {
        "task_id": args.task_id,
        "calibration_anchors": {"replayer_raw": anchor_low, "oracle_raw": anchor_high},
        "models": {},
        "separation": {},
    }

    print(f"=== Action sensitivity recompute (task_id={args.task_id}) ===")
    print(f"Controls anchors: replayer={anchor_low:.4f}, oracle={anchor_high:.4f} per m\n")

    for model_key, (csv_path, traj_dir) in datasets.items():
        if not csv_path.is_file():
            print(f"SKIP {model_key}: missing {csv_path.name}")
            continue
        rows = load_model_rows(csv_path, traj_dir, model_key)
        cx, cy, _ = __import__("analyze_check3", fromlist=["choose_center"]).choose_center(rows)
        summary = sens.position_sensitivity_summary(
            rows, cx, cy, anchor_low=anchor_low, anchor_high=anchor_high
        )
        wb = derived.get(model_key, {})
        results["models"][model_key] = {
            "n_valid": len(rows),
            "center_x": cx,
            "center_y": cy,
            **{k: v for k, v in summary.items() if k != "pairs"},
            "vla_trace_whitebox": wb,
        }
        print(
            f"{model_key}: raw={summary['raw_sensitivity_per_m']:.4f} "
            f"calibrated={summary['calibrated_0_1']:.3f} "
            f"(succ-only cal={summary['calibrated_success_only_0_1']:.3f}) "
            f"n_pairs={summary['n_pairs']}"
        )
        if wb:
            print(
                f"  VLA-Trace: visual_dep={wb.get('visual_dependency', float('nan')):.3f} "
                f"attn_mass={wb.get('attention_mass_object', float('nan')):.3f} "
                f"target_bg_drop={wb.get('patchmask_target_bg_avg_drop', float('nan')):.1f}"
            )

    openvla_cal = results["models"].get("openvla", {}).get("calibrated_0_1", float("nan"))
    pi05_cal = results["models"].get("pi05", {}).get("calibrated_0_1", float("nan"))
    openvla_raw = results["models"].get("openvla", {}).get("raw_sensitivity_per_m", float("nan"))
    pi05_raw = results["models"].get("pi05", {}).get("raw_sensitivity_per_m", float("nan"))
    openvla_vd = derived.get("openvla", {}).get("visual_dependency", float("nan"))
    pi05_vd = derived.get("pi05", {}).get("visual_dependency", float("nan"))
    openvla_am = derived.get("openvla", {}).get("attention_mass_object", float("nan"))
    pi05_am = derived.get("pi05", {}).get("attention_mass_object", float("nan"))

    action_delta = abs(pi05_cal - openvla_cal) if not (math.isnan(pi05_cal) or math.isnan(openvla_cal)) else float("nan")
    raw_delta = abs(pi05_raw - openvla_raw) if not (math.isnan(pi05_raw) or math.isnan(openvla_raw)) else float("nan")
    vd_delta = abs(pi05_vd - openvla_vd)
    am_delta = abs(pi05_am - openvla_am)

    results["separation"] = {
        "action_sensitivity_calibrated_delta": action_delta,
        "action_sensitivity_raw_delta": raw_delta,
        "vla_trace_visual_dependency_delta": vd_delta,
        "vla_trace_attention_mass_delta": am_delta,
        "action_sensitivity_separates_more_than_visual_dep": raw_delta > vd_delta,
        "action_sensitivity_separates_more_than_attn_mass": raw_delta > am_delta,
    }

    out_json = ROOT / (f"action_sensitivity_task{args.task_id}.json" if args.task_id else "action_sensitivity_task0.json")
    if args.task_id == 0:
        out_json = ROOT / "action_sensitivity_task0.json"
    out_json.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out_json.name}")
    print(
        f"\nSeparation: raw Δ={raw_delta:.4f} cal Δ={action_delta:.3f} vs "
        f"VLA-Trace visual_dep Δ={vd_delta:.3f} attn_mass Δ={am_delta:.3f}"
    )
    if results["separation"]["action_sensitivity_separates_more_than_visual_dep"]:
        print("FINDING: action-sensitivity separates models MORE than white-box visual dependency.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
