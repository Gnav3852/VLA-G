#!/usr/bin/env python3
"""Live PatchMask validation: correlate independent action-shift vs VLA-Trace ΔSR."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

import per_condition_correlation as pcc
import vla_trace_ground_truth as vtg
from patchmask_live_metrics import bootstrap_pearson_ci

ROOT = Path(__file__).resolve().parent
LIVE_SCORES = ROOT / "patchmask_behavioral_scores_live.json"
OUT_JSON = ROOT / "live_validation_correlation.json"
OUT_MD = ROOT / "LIVE_VALIDATION.md"

MODELS_LIVE = ("openvla", "pi05", "openvla_oft", "x_vla")


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    return pcc.spearman(x, y)


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    return pcc.pearson(x, y)


def align_live(white_box: list[dict[str, Any]], behavioral: list[dict[str, Any]]) -> list[dict[str, Any]]:
    beh_ok = [b for b in behavioral if b.get("status") == "ok" and b.get("model") in MODELS_LIVE]
    aligned = pcc.align_conditions(white_box, beh_ok)
    for row in aligned:
        beh = next(
            b
            for b in beh_ok
            if (b["model"], b["variant"], b["mode"], b["suite"])
            == (row["model"], row["variant"], row["mode"], row["suite"])
        )
        row["raw_l2_shift"] = beh.get("raw_l2_shift")
        row["normalized_l2_shift"] = beh.get("normalized_l2_shift")
        row["n_rollouts"] = beh.get("n_rollouts")
        row["behavioral_source"] = beh.get("source", "live_rollout")
    return aligned


def correlation_block(aligned: list[dict[str, Any]], y_key: str) -> dict[str, Any]:
    x = np.asarray([r["delta_sr"] for r in aligned], dtype=np.float64)
    y = np.asarray([r[y_key] for r in aligned], dtype=np.float64)
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    r = _pearson(x, y)
    rho = _spearman(x, y)
    ci = bootstrap_pearson_ci(x, y)
    return {
        "n": int(len(x)),
        "pearson_r": r,
        "spearman_rho": rho,
        "pearson_ci_95": ci,
        "small_n_flag": len(x) < 10,
        "y_metric": y_key,
    }


def within_model_correlation(aligned: list[dict[str, Any]], y_key: str) -> dict[str, Any]:
    by_model: dict[str, list[dict[str, Any]]] = {}
    for row in aligned:
        by_model.setdefault(row["model"], []).append(row)

    per_model = {}
    for model, rows in sorted(by_model.items()):
        if len(rows) < 3:
            per_model[model] = {"n": len(rows), "pearson_r": float("nan"), "spearman_rho": float("nan")}
            continue
        x = np.asarray([r["delta_sr"] for r in rows], dtype=np.float64)
        y = np.asarray([r[y_key] for r in rows], dtype=np.float64)
        per_model[model] = {
            "n": len(rows),
            "pearson_r": _pearson(x, y),
            "spearman_rho": _spearman(x, y),
        }
    return {"per_model": per_model, "n_models": len(by_model)}


def mask_contrast(aligned: list[dict[str, Any]], y_key: str) -> dict[str, Any]:
    """Compare target-black vs background-black action shift per model."""
    out: dict[str, Any] = {}
    for model in MODELS_LIVE:
        target = next(
            (r for r in aligned if r["model"] == model and r["variant"] == "mask_target" and r["mode"] == "black"),
            None,
        )
        bg = next(
            (r for r in aligned if r["model"] == model and r["variant"] == "mask_background" and r["mode"] == "black"),
            None,
        )
        if not target or not bg:
            out[model] = {"status": "missing_condition"}
            continue
        ty = float(target[y_key])
        by = float(bg[y_key])
        out[model] = {
            "target_black": ty,
            "background_black": by,
            "ratio_target_over_bg": ty / by if by > 1e-9 else float("inf"),
            "target_gt_background": ty > by,
        }
    return out


def magnitude_sensitivity_note(aligned: list[dict[str, Any]]) -> dict[str, Any]:
    raw = np.asarray([r.get("raw_l2_shift", float("nan")) for r in aligned], dtype=np.float64)
    norm = np.asarray([r.get("normalized_l2_shift", float("nan")) for r in aligned], dtype=np.float64)
    mask = ~(np.isnan(raw) | np.isnan(norm))
    if mask.sum() < 3:
        return {"note": "insufficient data"}
    r_raw_norm = _pearson(raw[mask], norm[mask])
    return {
        "raw_vs_normalized_pearson": r_raw_norm,
        "interpretation": (
            "High raw-vs-normalized agreement suggests metric tracks relative shift, not just action magnitude. "
            "If raw and normalized correlations with ΔSR diverge, prefer normalized for grounding claims."
        ),
    }


def write_markdown(report: dict[str, Any]) -> None:
    pooled_raw = report["pooled_raw_l2"]
    pooled_norm = report["pooled_normalized_l2"]
    lines = [
        "# Live PatchMask Validation",
        "",
        "**Proxy correlation (Pearson ~0.74 / Spearman ~0.88) is RETRACTED** — see [CIRCULARITY_AUDIT.md](CIRCULARITY_AUDIT.md).",
        "",
        "This report uses **live** paired masked/unmasked action-shift scores only.",
        "",
        "## Data",
        f"- Source: `{LIVE_SCORES.name}`",
        f"- Models: {', '.join(MODELS_LIVE)}",
        f"- Aligned conditions: {report['n_aligned']}",
        "",
        "## Pooled correlation (action shift vs VLA-Trace ΔSR)",
        "",
        "| Metric | n | Pearson r | 95% CI | Spearman ρ | small-n |",
        "|--------|---|-----------|--------|------------|---------|",
        f"| raw L2 shift | {pooled_raw['n']} | {pooled_raw['pearson_r']:.3f} | "
        f"[{pooled_raw['pearson_ci_95']['lo']:.3f}, {pooled_raw['pearson_ci_95']['hi']:.3f}] | "
        f"{pooled_raw['spearman_rho']:.3f} | {'yes' if pooled_raw['small_n_flag'] else 'no'} |",
        f"| normalized L2 shift | {pooled_norm['n']} | {pooled_norm['pearson_r']:.3f} | "
        f"[{pooled_norm['pearson_ci_95']['lo']:.3f}, {pooled_norm['pearson_ci_95']['hi']:.3f}] | "
        f"{pooled_norm['spearman_rho']:.3f} | {'yes' if pooled_norm['small_n_flag'] else 'no'} |",
        "",
        "## Within-model correlation (stronger claim)",
        "",
    ]
    for model, stats in report["within_model_raw"]["per_model"].items():
        lines.append(
            f"- **{model}** (n={stats['n']}): Pearson r={stats['pearson_r']:.3f}, "
            f"Spearman ρ={stats['spearman_rho']:.3f}"
        )
    lines.extend(["", "## Target vs background mask contrast (black)", ""])
    for model, c in report["mask_contrast_raw"].items():
        if c.get("status") == "missing_condition":
            lines.append(f"- **{model}**: missing target or background condition")
        else:
            lines.append(
                f"- **{model}**: target={c['target_black']:.4f}, background={c['background_black']:.4f}, "
                f"ratio={c['ratio_target_over_bg']:.2f}, target>background={c['target_gt_background']}"
            )
    lines.extend(
        [
            "",
            "## Magnitude sensitivity",
            "",
            f"- Raw vs normalized shift Pearson: {report['magnitude_note'].get('raw_vs_normalized_pearson', float('nan')):.3f}",
            f"- {report['magnitude_note'].get('interpretation', '')}",
            "",
            "## Honest read",
            "",
            report["interpretation"],
            "",
        ]
    )
    OUT_MD.write_text("\n".join(lines))


def main() -> None:
    if not LIVE_SCORES.is_file():
        raise SystemExit(f"Missing {LIVE_SCORES}; run modal_patchmask_probe.py --use-proxy=false first")

    live = json.loads(LIVE_SCORES.read_text())
    behavioral = live.get("conditions", [])
    sanity = live.get("sanity_gate") or {}

    wb = [r for r in pcc.white_box_conditions() if r["model"] in MODELS_LIVE and r["suite"] == "libero_spatial"]
    aligned = align_live(wb, behavioral)

    contrast_raw = mask_contrast(aligned, "raw_l2_shift")
    models_passing = [m for m, s in sanity.get("per_model", {}).items() if s.get("pass")]
    if not models_passing:
        report = {
            "n_aligned": len(aligned),
            "sanity_gate": sanity,
            "mask_contrast_raw": contrast_raw,
            "blocked": True,
            "interpretation": (
                "Correlation BLOCKED: no model passed sanity gate (visual target placement + "
                "normalized target shift). Inspect patchmask_frames/ and mask_verification stats."
            ),
        }
        OUT_JSON.write_text(json.dumps(report, indent=2, default=lambda o: float(o) if isinstance(o, (np.floating, float)) and math.isnan(o) else o))
        lines = [
            "# Live PatchMask Validation",
            "",
            "**Status: BLOCKED — sanity gate failed**",
            "",
            "Target-black action shift must exceed background-black before reporting correlation.",
            "",
            "## Sanity gate",
            "",
        ]
        for model, stats in sanity.get("per_model", {}).items():
            lines.append(f"- **{model}**: {stats}")
        lines.extend(["", report["interpretation"], ""])
        OUT_MD.write_text("\n".join(lines))
        print(f"Sanity gate failed — wrote blocker report to {OUT_MD}")
        raise SystemExit(1)

    # Restrict to models that passed sanity when some models failed
    if models_passing and len(models_passing) < len(MODELS_LIVE):
        aligned = [r for r in aligned if r["model"] in models_passing]
        behavioral = [b for b in behavioral if b.get("model") in models_passing]
        wb = [r for r in wb if r["model"] in models_passing]

    pooled_raw = correlation_block(aligned, "raw_l2_shift")
    pooled_norm = correlation_block(aligned, "normalized_l2_shift")
    within_raw = within_model_correlation(aligned, "raw_l2_shift")
    within_norm = within_model_correlation(aligned, "normalized_l2_shift")
    contrast_raw = mask_contrast(aligned, "raw_l2_shift")
    mag = magnitude_sensitivity_note(aligned)

    n_models = within_raw["n_models"]
    if pooled_raw["n"] < 5:
        interp = (
            "Too few aligned conditions for a stable pooled correlation. "
            "Treat any r as exploratory until more rollouts and mask types are collected."
        )
    elif n_models < 2:
        failed = [m for m in MODELS_LIVE if m not in models_passing]
        extra = f" Models blocked: {', '.join(failed)}." if failed else ""
        interp = (
            f"Only one model has passing live scores; pooled correlation is exploratory (n={pooled_raw['n']})."
            + extra
        )
    else:
        interp = (
            "Compare pooled vs within-model r: if within-model tracking is weak while pooled r looks strong, "
            "agreement may be driven by between-model 'general goodness' rather than per-condition alignment."
        )

    report = {
        "n_aligned": len(aligned),
        "sanity_gate": sanity,
        "models_passing_sanity": models_passing,
        "pooled_raw_l2": pooled_raw,
        "pooled_normalized_l2": pooled_norm,
        "within_model_raw": within_raw,
        "within_model_normalized": within_norm,
        "mask_contrast_raw": contrast_raw,
        "magnitude_note": mag,
        "interpretation": interp,
        "aligned_rows": aligned,
    }
    OUT_JSON.write_text(json.dumps(report, indent=2, default=lambda o: float(o) if isinstance(o, (np.floating, float)) and math.isnan(o) else o))
    write_markdown(report)
    print(f"Wrote {OUT_JSON} and {OUT_MD}")
    print(f"Pooled raw L2: r={pooled_raw['pearson_r']:.3f}, rho={pooled_raw['spearman_rho']:.3f}, n={pooled_raw['n']}")


if __name__ == "__main__":
    main()
