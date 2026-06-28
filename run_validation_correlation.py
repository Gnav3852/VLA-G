#!/usr/bin/env python3
"""
Phase 4-5: per-condition + per-model correlation vs VLA-Trace white-box numbers.

Reads:
  - vla_trace_numbers.json (white-box ground truth)
  - action_sensitivity_task*.json (position probe, per-model)
  - patchmask_behavioral_scores.json (occlusion probe, per-condition) if present

Writes COMPARISON_VALIDATION.md and validation_correlation.json

Run:
  ./.venv/bin/python run_validation_correlation.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

import per_condition_correlation as pcc
import vla_trace_ground_truth as vtg


def json_safe(obj: Any) -> Any:
    if isinstance(obj, float) and math.isnan(obj):
        return None
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [json_safe(v) for v in obj]
    return obj

ROOT = Path(__file__).resolve().parent


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text()) if path.is_file() else {}


def write_report(
    per_condition: dict[str, Any],
    per_model: dict[str, Any],
    position_results: dict[str, Any],
    out_md: Path,
) -> None:
    lines = [
        "# Validation: behavioral action-sensitivity vs VLA-Trace white-box",
        "",
        "## Per-model position sensitivity (Probe A)",
        "",
        "| model | raw sensitivity | calibrated 0-1 | VLA-Trace visual_dep | attn mass | target_bg drop |",
        "|-------|-----------------|----------------|----------------------|-----------|----------------|",
    ]
    for model, data in sorted(position_results.get("models", {}).items()):
        wb = data.get("vla_trace_whitebox", {})
        lines.append(
            f"| {model} | {data.get('raw_sensitivity_per_m', float('nan')):.4f} | "
            f"{data.get('calibrated_0_1', float('nan')):.3f} | "
            f"{wb.get('visual_dependency', float('nan')):.3f} | "
            f"{wb.get('attention_mass_object', float('nan')):.3f} | "
            f"{wb.get('patchmask_target_bg_avg_drop', float('nan')):.1f} |"
        )

    sep = position_results.get("separation", {})
    lines.extend(
        [
            "",
            f"**Separation test:** raw Δ={sep.get('action_sensitivity_raw_delta', float('nan')):.4f} "
            f"cal Δ={sep.get('action_sensitivity_calibrated_delta', float('nan')):.3f} "
            f"vs VLA-Trace visual_dep Δ={sep.get('vla_trace_visual_dependency_delta', float('nan')):.3f} "
            f"(attn_mass Δ={sep.get('vla_trace_attention_mass_delta', float('nan')):.3f}).",
            "",
            f"- Action sensitivity separates more than visual_dep: **{sep.get('action_sensitivity_separates_more_than_visual_dep')}**",
            f"- Action sensitivity separates more than attn mass: **{sep.get('action_sensitivity_separates_more_than_attn_mass')}**",
            "",
            "## Per-condition PatchMask correlation (Probe C)",
            "",
            f"- N aligned conditions: {per_condition.get('n', 0)}",
            f"- Pearson r: {per_condition.get('pearson_r', float('nan')):.4f}",
            f"- R²: {per_condition.get('r_squared', float('nan')):.4f}",
            f"- Spearman ρ: {per_condition.get('spearman_rho', float('nan')):.4f}",
            f"- Partial r (control baseline SR): {per_condition.get('partial_r_controlling_baseline', float('nan')):.4f}",
            "",
            "## Per-model rank sanity check",
            "",
            f"- Models: {', '.join(per_model.get('models', []))}",
            f"- Spearman ρ (position vs target_bg drop): {per_model.get('spearman_rho', float('nan')):.4f}",
            "",
            "## Interpretation",
            "",
        ]
    )

    if sep.get("action_sensitivity_separates_more_than_visual_dep"):
        lines.append(
            "Our action-sensitivity metric produces **larger pi0.5–OpenVLA separation** than "
            "VLA-Trace's near-tied white-box visual-dependency / attention-mass scalars on the same pair."
        )
    else:
        lines.append(
            "Action-sensitivity did **not** clearly outperform white-box separation on pi0.5 vs OpenVLA; "
            "interpret per-condition results cautiously."
        )

    if per_condition.get("n", 0) >= 10 and not math.isnan(per_condition.get("pearson_r", float("nan"))):
        r = per_condition["pearson_r"]
        if r > 0.5:
            lines.append(
                f"Per-condition PatchMask correlation is **positive** (r={r:.2f}): larger VLA-Trace SR drops "
                "align with larger behavioral action shifts."
            )
        else:
            lines.append(
                f"Per-condition correlation is **weak** (r={r:.2f}); may need more mask conditions or suites."
            )
    else:
        lines.append(
            "Per-condition correlation used available PatchMask behavioral scores; expand "
            "`patchmask_behavioral_scores.json` via `modal_patchmask_probe.py` for full coverage."
        )

    out_md.write_text("\n".join(lines) + "\n")
    print(f"Wrote {out_md}")


def plot_scatter(aligned: list[dict[str, Any]], out_png: Path) -> None:
    if not aligned:
        return
    x = [r["delta_sr"] for r in aligned]
    y = [r["action_sensitivity"] for r in aligned]
    plt.figure(figsize=(7, 5))
    plt.scatter(x, y, alpha=0.6, s=30)
    plt.xlabel("VLA-Trace ΔSR (baseline - masked)")
    plt.ylabel("Our action sensitivity (calibrated 0-1)")
    plt.title("Per-condition PatchMask: white-box vs behavioral")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()
    print(f"Wrote {out_png}")


def main() -> int:
    position_path = ROOT / "action_sensitivity_task0.json"
    patchmask_path = ROOT / "patchmask_behavioral_scores.json"
    position_results = load_json(position_path)
    behavioral = pcc.load_behavioral_scores(patchmask_path)

    white_box = pcc.white_box_conditions()
    aligned = pcc.align_conditions(white_box, behavioral)
    per_condition = pcc.correlation_report(aligned)

    position_by_model = {
        m: d.get("raw_sensitivity_per_m", float("nan")) for m, d in position_results.get("models", {}).items()
    }
    per_model = pcc.per_model_rank_correlation(position_by_model)

    out = {
        "per_condition": per_condition,
        "per_model_rank": per_model,
        "n_white_box_conditions": len(white_box),
        "n_aligned_conditions": len(aligned),
        "position_separation": position_results.get("separation", {}),
        "derived_per_model_whitebox": vtg.derived_per_model(),
    }
    out_json = ROOT / "validation_correlation.json"
    out_json.write_text(json.dumps(json_safe(out), indent=2))
    print(f"Wrote {out_json.name}")

    write_report(per_condition, per_model, position_results, ROOT / "COMPARISON_VALIDATION.md")
    plot_scatter(aligned, ROOT / "validation_per_condition_scatter.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
