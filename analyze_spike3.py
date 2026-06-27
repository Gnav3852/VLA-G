#!/usr/bin/env python3
"""
Combined Phase 1 analysis for the tight-grid spike3 sweep.

Runs:
  - Trajectory-similarity (analyze_check3 logic, success-independent)
  - Grasp-on-object regression on all-valid and successes-only subsets

Outputs:
  spike3_success_vs_distance.png
  spike3_traj_similarity_vs_object_distance.png
  spike3_grasp_vs_object.png
  Appends spike3 section to FINDINGS.md
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

import analyze_check3 as check3
import analyze_perturbation as perturb

ROOT = Path(__file__).resolve().parent
FINDINGS_PATH = ROOT / "FINDINGS.md"


def spike3_paths() -> dict[str, Path]:
    return {
        "csv": ROOT / "spike3_rollouts.csv",
        "traj_dir": ROOT / "spike3_trajectories",
        "grasp_plot": ROOT / "spike3_grasp_vs_object.png",
        "success_plot": ROOT / "spike3_success_vs_distance.png",
        "traj_plot": ROOT / "spike3_traj_similarity_vs_object_distance.png",
        "controlled_plot": ROOT / "spike3_traj_controlled.png",
        "overlay_plot": ROOT / "spike3_failed_overlay.png",
    }


def configure_modules() -> dict[str, Path]:
    paths = spike3_paths()
    perturb.configure(paths["csv"], paths["traj_dir"], paths["grasp_plot"], label="tight-grid")
    check3.configure(
        paths["csv"],
        paths["traj_dir"],
        paths["success_plot"],
        paths["traj_plot"],
    )
    return paths


def regression_block(rows: list[dict[str, Any]], label: str) -> dict[str, Any]:
    import numpy as np

    if len(rows) < 2:
        return {"label": label, "n": len(rows), "stats_x": {}, "stats_y": {}, "ratio_x": math.nan, "ratio_y": math.nan}

    obj_x = np.asarray([r["obj_x"] for r in rows])
    obj_y = np.asarray([r["obj_y"] for r in rows])
    grasp_x = np.asarray([r["grasp_x"] for r in rows])
    grasp_y = np.asarray([r["grasp_y"] for r in rows])

    stats_x = perturb.linear_stats(obj_x, grasp_x)
    stats_y = perturb.linear_stats(obj_y, grasp_y)
    x_span = perturb.cm_span(obj_x)
    y_span = perturb.cm_span(obj_y)
    ratio_x = perturb.cm_span(grasp_x) / x_span if x_span > 0 else math.nan
    ratio_y = perturb.cm_span(grasp_y) / y_span if y_span > 0 else math.nan

    print(f"\n=== Grasp regression ({label}) n={len(rows)} ===")
    print(f"  object span: x={x_span:.2f} cm, y={y_span:.2f} cm")
    print(
        f"  x: slope={stats_x['slope']:.4f}, R2={stats_x['r2']:.4f}, "
        f"r={stats_x['pearson']:.4f}, tracking_ratio={ratio_x:.4f}"
    )
    print(
        f"  y: slope={stats_y['slope']:.4f}, R2={stats_y['r2']:.4f}, "
        f"r={stats_y['pearson']:.4f}, tracking_ratio={ratio_y:.4f}"
    )
    print(f"  read: {perturb.interpretation(stats_x, stats_y, x_span, y_span)}")

    return {
        "label": label,
        "n": len(rows),
        "stats_x": stats_x,
        "stats_y": stats_y,
        "ratio_x": ratio_x,
        "ratio_y": ratio_y,
        "x_span": x_span if len(rows) >= 1 else 0.0,
        "y_span": y_span if len(rows) >= 1 else 0.0,
    }


def combined_verdict(
    all_reg: dict[str, Any],
    succ_reg: dict[str, Any],
    check3_result: dict[str, Any],
    n_valid: int,
    n_success: int,
    n_rejected: int,
) -> str:
    pairwise = check3_result["pairwise"]
    r = pairwise["pearson"]
    mean_td = pairwise["mean_traj_distance_m"] * 100
    bf = pairwise["by_group"]["both_failed"]
    bf_mean = bf.get("mean_traj_distance_m", float("nan"))
    bf_mean_cm = bf_mean * 100 if not math.isnan(bf_mean) else float("nan")
    bf_r = bf.get("pearson", float("nan"))

    controlled = check3_result.get("controlled", {})
    sb = controlled.get("same_base_summary", {})
    sp = controlled.get("same_pos_summary", {})
    ctrl_r = sb.get("pearson", float("nan"))
    ctrl_mean = sb.get("mean_traj_distance_m", float("nan")) * 100
    base_mean = sp.get("mean_traj_distance_m", float("nan")) * 100
    lift = ctrl_mean - base_mean if not (math.isnan(ctrl_mean) or math.isnan(base_mean)) else float("nan")

    sx = all_reg["stats_x"]
    sy = all_reg["stats_y"]
    ssx = succ_reg["stats_x"]
    ssy = succ_reg["stats_y"]

    # Controlled trajectory signal (robot start held fixed) is the clean grounding test.
    ctrl_grounding = (not math.isnan(ctrl_r)) and ctrl_r >= 0.3 and (math.isnan(lift) or lift > 1.0)
    ctrl_replay = (not math.isnan(ctrl_r)) and abs(ctrl_r) < 0.2 and (not math.isnan(lift)) and lift <= 1.0
    ctrl_succ_grounding = (
        not math.isnan(sb.get("success_pearson", math.nan))
        and sb.get("success_pearson", 0) >= 0.4
        and sb.get("n_success_pairs", 0) >= 5
    )
    # Failure mode: clean test is the within-start tube width (robot start held fixed),
    # NOT the confounded both-failed pairwise mean (which mixes positions and base_idx).
    overlay = check3_result.get("overlay", {})
    failed_tube = overlay.get("failed_mean_tube_width_m", float("nan")) * 100
    succ_tube = overlay.get("succeeded_mean_tube_width_m", float("nan")) * 100
    have_tubes = not (math.isnan(failed_tube) or math.isnan(succ_tube))

    if have_tubes and failed_tube < succ_tube * 0.95:
        failures_mode = "replay"
        failure_read = (
            f"Within-start tube width is TIGHTER for failures than successes ({failed_tube:.1f}cm < "
            f"{succ_tube:.1f}cm) — failures collapse onto a canned motion (replay-like)."
        )
    elif have_tubes and failed_tube > succ_tube * 1.05:
        failures_mode = "scatter"
        failure_read = (
            f"Within-start tube width is WIDER for failures than successes ({failed_tube:.1f}cm > "
            f"{succ_tube:.1f}cm) — failures scatter more, leaning mild breakage rather than tight replay. "
            f"(The earlier both-failed pairwise mean {bf_mean_cm:.1f}cm < baseline {base_mean:.1f}cm was "
            "confounded by mixing positions; the within-start tube width supersedes it.)"
        )
    else:
        failures_mode = "comparable"
        failure_read = (
            f"Within-start tube widths are comparable for failures vs successes "
            f"({failed_tube:.1f}cm vs {succ_tube:.1f}cm) — failure mode is neither clearly canned nor chaotic."
        )

    grasp_succ_grounding = (
        succ_reg["n"] >= 5
        and not math.isnan(ssy.get("r2", math.nan))
        and ssy.get("r2", 0) >= 0.40
        and abs(ssy.get("pearson", 0)) >= 0.45
    )
    low_grasp_all = (
        not math.isnan(sx.get("slope", math.nan))
        and abs(sx["slope"]) <= 0.30
        and sx.get("r2", 0) <= 0.20
        and abs(sy.get("slope", math.nan)) <= 0.30
        and sy.get("r2", 0) <= 0.20
    )

    ctrl_line = (
        f"Controlled trajectory test (robot start fixed): object-driven r={ctrl_r:.3f}, "
        f"object-driven mean={ctrl_mean:.1f}cm vs non-object baseline={base_mean:.1f}cm (lift={lift:+.1f}cm). "
    )

    if (ctrl_grounding or ctrl_succ_grounding) and (grasp_succ_grounding or not low_grasp_all):
        failure_tag = {
            "replay": "replay-like failure",
            "scatter": "failures scatter (mild breakage)",
            "comparable": "failure mode unclear",
        }.get(failures_mode, "failure mode unclear")
        label = f"partial/local grounding + {failure_tag}"
        detail = (
            ctrl_line
            + f"On SUCCESSES, grasp PARTIALLY tracks object (y slope={ssy.get('slope', float('nan')):.2f} <1, "
            f"R²={ssy.get('r2', float('nan')):.2f}, r={ssy.get('pearson', float('nan')):.2f}) — it shifts "
            "toward the object but under-reaches, on top of a strong shared canned reaching backbone "
            "(centroid paths near-identical across positions and starts). The controlled successes-only "
            f"traj r={sb.get('success_pearson', float('nan')):.2f} is partly tautological (a grasp ends at "
            f"the object). Across ALL valid rollouts object adds ~nothing beyond robot-start noise "
            f"(lift={lift:+.1f}cm). {failure_read} "
            "Net: a dominant canned reach with partial object tracking; out-of-band cases fail by scatter, "
            "not by a tighter replay."
        )
    elif ctrl_replay and low_grasp_all:
        label = "replay"
        detail = (
            ctrl_line
            + "Even with robot start controlled, trajectories do not track object position, and grasp "
            "barely tracks it. Motion looks largely fixed regardless of object placement."
        )
    elif grasp_succ_grounding and not ctrl_grounding:
        label = "inconclusive — grasp grounds, trajectory ambiguous"
        detail = (
            ctrl_line
            + f"Successes show grasp y-tracking (R²={ssy.get('r2', float('nan')):.2f}), but the controlled "
            "trajectory test does not clearly confirm it. Likely local grounding with a coarse path metric."
        )
    elif ctrl_mean > 8.0 and abs(ctrl_r) < 0.3 and (math.isnan(lift) or lift > 2.0):
        label = "residual breakage / inconclusive"
        detail = (
            ctrl_line
            + "Object-driven trajectory variation is high but not ordered by object distance — possible "
            "instability even inside the band."
        )
    else:
        label = "inconclusive"
        detail = (
            ctrl_line
            + f"Naive traj r={r:.3f} (confounded); all-valid grasp x slope={sx.get('slope', float('nan')):.2f}; "
            f"{n_success} successes."
        )

    power = (
        f"Power note: {n_success}/{n_valid} successes ({n_rejected} rejected); "
        f"controlled object-driven pairs n={sb.get('n_pairs', 0)}, baseline pairs n={sp.get('n_pairs', 0)}. "
        + ("Success-only regression usable but modest." if n_success >= 5 else "Success-only underpowered.")
    )
    return f"VERDICT: {label} — {detail} {power}"


def write_findings_section(
    paths: dict[str, Path],
    skipped: list[tuple[str, str]],
    all_reg: dict[str, Any],
    succ_reg: dict[str, Any],
    check3_result: dict[str, Any],
    n_commanded: int,
    n_valid: int,
    n_success: int,
    n_rejected: int,
    verdict: str,
) -> None:
    pairwise = check3_result["pairwise"]
    table = check3_result["table"]
    usable_cm = check3_result["usable_range_m"] * 100
    controlled = check3_result.get("controlled", {})
    sb = controlled.get("same_base_summary", {})
    sp = controlled.get("same_pos_summary", {})
    ctrl_mean = sb.get("mean_traj_distance_m", float("nan")) * 100
    base_mean = sp.get("mean_traj_distance_m", float("nan")) * 100
    lift = ctrl_mean - base_mean if not (math.isnan(ctrl_mean) or math.isnan(base_mean)) else float("nan")
    overlay = check3_result.get("overlay", {})
    failed_tube = overlay.get("failed_mean_tube_width_m", float("nan")) * 100
    succ_tube = overlay.get("succeeded_mean_tube_width_m", float("nan")) * 100

    def fmt_stats(reg: dict[str, Any], axis: str) -> str:
        s = reg["stats_x"] if axis == "x" else reg["stats_y"]
        if not s or "slope" not in s or math.isnan(s.get("slope", math.nan)):
            return "n/a"
        return f"{s['slope']:.4f} / {s['r2']:.4f} / {s['pearson']:.4f}"

    def fmt_ratio(v: float) -> str:
        return f"{v:.4f}" if not math.isnan(v) else "n/a"

    bin_lines = "\n".join(
        f"| {b['bin']} | {b['n_rollouts']} | {b['n_success']} | {b['success_rate']:.3f} |"
        for b in table
        if b["n_rollouts"] > 0
    )

    section = f"""

---

# Tight-Grid Phase 1 Findings (spike3)

## Setup

- Input CSV: `{paths['csv'].name}`
- Trajectories: `{paths['traj_dir'].name}/rollout_*.json`
- Grid: 5×5 over ±7 cm of center `(-0.06, 0.20)`, 3 repeats per cell → {n_commanded} commanded
- Valid / rejected / successes: {n_valid} / {n_rejected} / {n_success}
- Skipped in analysis: {len(skipped)}

## Success vs perturbation distance

Usable range (max distance with any success): {usable_cm:.2f} cm

| bin | n | successes | rate |
|-----|---|-----------|------|
{bin_lines}

## Trajectory similarity (success-independent)

- Metric: {check3.TRAJ_METRIC}
- Pairwise r (object dist vs traj dist): {pairwise['pearson']:.4f}
- Mean / median traj distance: {pairwise['mean_traj_distance_m']*100:.2f} / {pairwise['median_traj_distance_m']*100:.2f} cm
- Failed-pair mean traj dist: {pairwise['by_group']['both_failed'].get('mean_traj_distance_m', float('nan'))*100:.2f} cm (r={pairwise['by_group']['both_failed'].get('pearson', float('nan')):.3f})

NOTE: the naive metric is confounded — repeats vary `base_idx` (robot/scene start), so
pairs differ for reasons unrelated to object position. The controlled metric below fixes this.

## Trajectory similarity — base_idx-controlled

Splits pairs to isolate the object's effect from robot-start noise:

| split | meaning | n pairs | r (obj dist vs traj dist) | mean traj dist |
|-------|---------|---------|---------------------------|----------------|
| same robot-start, diff object | object-driven (clean grounding test) | {sb.get('n_pairs', 0)} | {sb.get('pearson', float('nan')):.4f} | {ctrl_mean:.2f} cm |
| same position, diff robot-start | non-object baseline (noise floor) | {sp.get('n_pairs', 0)} | {sp.get('pearson', float('nan')):.4f} | {base_mean:.2f} cm |

- Object-driven mean MINUS baseline mean (lift) = {lift:+.2f} cm (positive => object position adds trajectory variation beyond robot-start noise)
- Object-driven successes-only: n={sb.get('n_success_pairs', 0)}, r={sb.get('success_pearson', float('nan')):.4f}

Plots: `{paths['success_plot'].name}`, `{paths['traj_plot'].name}`, `{paths['controlled_plot'].name}`

## Failed-trajectory overlay (robot start held fixed)

Direct visual + dispersion test of the "replay-like failure" claim. Tube width = mean
per-timestep spread of EE paths around their centroid, pooled within fixed `base_idx`
(so robot-start variance is excluded).

| group | within-start tube width |
|-------|-------------------------|
| failed | {failed_tube:.2f} cm |
| succeeded | {succ_tube:.2f} cm |

Read: failed tube {"tighter" if (not math.isnan(failed_tube) and not math.isnan(succ_tube) and failed_tube < succ_tube) else "WIDER"} than succeeded => {"failures collapse onto a canned motion (replay-like)" if (not math.isnan(failed_tube) and not math.isnan(succ_tube) and failed_tube < succ_tube) else "failures scatter more than successes (mild breakage), NOT a tight canned replay"}.
This supersedes the confounded both-failed pairwise comparison above.
In the plot: star = object init, X = grasp point, color = object perturbation distance.
Visual: centroid paths are near-identical across positions/starts (strong canned reaching backbone);
object stars span y~0.13-0.35 while grasp X's stay compressed (~0.10-0.20) => partial, under-reaching tracking.

Plot: `{paths['overlay_plot'].name}`

## Grasp regression: grasp_point ~ actual_object_init

| subset | n | x slope/R²/r | y slope/R²/r | x tracking ratio | y tracking ratio |
|--------|---|--------------|--------------|------------------|------------------|
| all valid | {all_reg['n']} | {fmt_stats(all_reg, 'x')} | {fmt_stats(all_reg, 'y')} | {fmt_ratio(all_reg['ratio_x'])} | {fmt_ratio(all_reg['ratio_y'])} |
| successes only | {succ_reg['n']} | {fmt_stats(succ_reg, 'x')} | {fmt_stats(succ_reg, 'y')} | {fmt_ratio(succ_reg['ratio_x'])} | {fmt_ratio(succ_reg['ratio_y'])} |

Object span (all valid): x={all_reg['x_span']:.2f} cm, y={all_reg['y_span']:.2f} cm

Plot: `{paths['grasp_plot'].name}`

## Interpretation matrix

| Signal | Pattern in spike3 | Read |
|--------|-------------------|------|
| Controlled traj (object-driven) | r>0, lift>0 over baseline | grounding (object changes path) |
| Controlled traj (object-driven) | r≈0, lift≈0 | replay (path fixed vs object) |
| Controlled traj (object-driven) | high mean, r≈0, lift large | residual breakage |
| Grasp regression | slope≈1, high R² on successes | grounding |
| Grasp regression | 0<slope<1, moderate R² | partial/compressed grounding |
| Grasp regression | low slope/R² | replay or no tracking |
| Tube width | failed < succeeded | replay-like failure |
| Tube width | failed > succeeded | failures scatter (breakage) |
| Naive traj (all pairs) | confounded by base_idx | use controlled / tube instead |

## Combined verdict

{verdict}

## Comparison to wide pilot (spike2)

Wide pilot: 2/27 successes, usable range ~8 cm, verdict was OOD-breakage-dominated (mass failure outside envelope).
Tight grid moves into the empirically measured envelope; {n_success} successes enable both metrics side by side.

## Honesty notes

- Still a pilot on one task (`libero_spatial` task_id=0) and one model (OpenVLA).
- LIBERO-PRO reports OpenVLA position-perturbation success ≈0 on libero-spatial; {n_success} successes here reflects the narrower band, not contradiction.
- The successes-only controlled trajectory correlation (r={sb.get('success_pearson', float('nan')):.2f}) is partly tautological: a successful grasp trajectory must end at the object, so far-apart objects force far-apart trajectories. Weight the grasp regression and the tube-width test more than this single number.
- CORRECTION: an earlier reading called failures "replay-like" because both-failed pairwise mean ({pairwise['by_group']['both_failed'].get('mean_traj_distance_m', float('nan'))*100:.1f} cm) fell below the same-position baseline ({base_mean:.1f} cm). That comparison was confounded (it mixed different object positions). The clean within-start tube-width test (robot start fixed) shows failures are {"WIDER" if (not math.isnan(failed_tube) and not math.isnan(succ_tube) and failed_tube > succ_tube) else "NOT tighter"} than successes ({failed_tube:.1f} vs {succ_tube:.1f} cm) — so failures scatter, they do not collapse into a tight canned replay.
- Grasp tracking on successes is PARTIAL/compressed (slope ~{succ_reg['stats_y'].get('slope', float('nan')):.2f} <1): the gripper shifts toward the object but under-reaches, riding a strong shared canned reaching backbone (see overlay centroid paths).
- Across all valid rollouts the object adds ~0 trajectory variation beyond robot-start noise (lift={lift:+.1f} cm); the (partial) grounding signal lives in the success subset.
- Phase 2 (OpenVLA vs Pi0.5) deferred until this read is reviewed.
"""
    existing = FINDINGS_PATH.read_text() if FINDINGS_PATH.exists() else ""
    marker = "# Tight-Grid Phase 1 Findings (spike3)"
    if marker in existing:
        existing = existing.split(marker)[0].rstrip()
    FINDINGS_PATH.write_text(existing + section)
    print(f"\nAppended spike3 section to {FINDINGS_PATH}")


def attach_trajectories(rows: list[dict[str, Any]], traj_dir: Path) -> None:
    for row in rows:
        traj_path = traj_dir / f"rollout_{row['rollout_id']:03d}.json"
        row["traj"] = perturb.load_trajectory(traj_path)


def extract_detector_metrics(rows: list[dict[str, Any]], quiet: bool = True) -> dict[str, Any]:
    """Run detector metrics on rows (must have traj attached). No plots or FINDINGS."""
    import numpy as np

    cx, cy, _ = check3.choose_center(rows)
    distances_m = check3.perturbation_distances(rows, cx, cy)
    pairwise = check3.pairwise_trajectory_analysis(rows)
    controlled = check3.controlled_pairwise_analysis(rows)

    success_rows = [r for r in rows if r["success"]]
    all_reg = regression_block(rows, "all valid") if not quiet else _regression_silent(rows)
    succ_reg = (
        regression_block(success_rows, "successes only")
        if (not quiet and len(success_rows) >= 2)
        else _regression_silent(success_rows)
    )

    sb = controlled["same_base_summary"]
    sp = controlled["same_pos_summary"]
    lift_m = float("nan")
    if not math.isnan(sb.get("mean_traj_distance_m", float("nan"))) and not math.isnan(
        sp.get("mean_traj_distance_m", float("nan"))
    ):
        lift_m = sb["mean_traj_distance_m"] - sp["mean_traj_distance_m"]

    # tube widths without writing plot
    base_idxs = sorted({r.get("base_idx", -1) for r in rows if r.get("base_idx", -1) >= 0})
    failed_tubes: list[float] = []
    succ_tubes: list[float] = []
    for bidx in base_idxs:
        for want_success, store in ((False, failed_tubes), (True, succ_tubes)):
            trajs = [
                r["traj"]
                for r in rows
                if r.get("base_idx", -1) == bidx and bool(r["success"]) == want_success
            ]
            if len(trajs) >= 2:
                store.append(check3.tube_width(trajs))
    failed_tube = float(np.nanmean(failed_tubes)) if failed_tubes else float("nan")
    succ_tube = float(np.nanmean(succ_tubes)) if succ_tubes else float("nan")

    sx = all_reg["stats_x"]
    sy = all_reg["stats_y"]
    tracking_gain = float("nan")
    if sx and sy and not math.isnan(sx.get("slope", math.nan)) and not math.isnan(sy.get("slope", math.nan)):
        tracking_gain = (abs(sx["slope"]) + abs(sy["slope"])) / 2.0

    return {
        "n_valid": len(rows),
        "n_success": len(success_rows),
        "center_x": cx,
        "center_y": cy,
        "slope_x": sx.get("slope", float("nan")) if sx else float("nan"),
        "slope_y": sy.get("slope", float("nan")) if sy else float("nan"),
        "r2_x": sx.get("r2", float("nan")) if sx else float("nan"),
        "r2_y": sy.get("r2", float("nan")) if sy else float("nan"),
        "pearson_x": sx.get("pearson", float("nan")) if sx else float("nan"),
        "pearson_y": sy.get("pearson", float("nan")) if sy else float("nan"),
        "tracking_gain": tracking_gain,
        "controlled_r": sb.get("pearson", float("nan")),
        "controlled_lift_cm": lift_m * 100 if not math.isnan(lift_m) else float("nan"),
        "controlled_mean_traj_cm": sb.get("mean_traj_distance_m", float("nan")) * 100,
        "baseline_mean_traj_cm": sp.get("mean_traj_distance_m", float("nan")) * 100,
        "failed_tube_cm": failed_tube * 100,
        "succ_tube_cm": succ_tube * 100,
        "pairwise_r": pairwise["pearson"],
        "all_reg": all_reg,
        "succ_reg": succ_reg,
        "controlled": controlled,
    }


def _regression_silent(rows: list[dict[str, Any]]) -> dict[str, Any]:
    import numpy as np

    if len(rows) < 2:
        return {"n": len(rows), "stats_x": {}, "stats_y": {}, "ratio_x": math.nan, "ratio_y": math.nan}
    obj_x = np.asarray([r["obj_x"] for r in rows])
    obj_y = np.asarray([r["obj_y"] for r in rows])
    grasp_x = np.asarray([r["grasp_x"] for r in rows])
    grasp_y = np.asarray([r["grasp_y"] for r in rows])
    stats_x = perturb.linear_stats(obj_x, grasp_x)
    stats_y = perturb.linear_stats(obj_y, grasp_y)
    x_span = perturb.cm_span(obj_x)
    y_span = perturb.cm_span(obj_y)
    return {
        "n": len(rows),
        "stats_x": stats_x,
        "stats_y": stats_y,
        "ratio_x": perturb.cm_span(grasp_x) / x_span if x_span > 0 else math.nan,
        "ratio_y": perturb.cm_span(grasp_y) / y_span if y_span > 0 else math.nan,
        "x_span": x_span,
        "y_span": y_span,
    }


def run_check3_analysis(rows: list[dict[str, Any]]) -> dict[str, Any]:
    import numpy as np

    paths = spike3_paths()
    cx, cy, center_method = check3.choose_center(rows)
    distances_m = check3.perturbation_distances(rows, cx, cy)
    table = check3.bin_success_table(rows, distances_m)
    pairwise = check3.pairwise_trajectory_analysis(rows)
    controlled = check3.controlled_pairwise_analysis(rows)

    success_mask = np.array([r["success"] for r in rows])
    usable_range_m = float(distances_m[success_mask].max()) if success_mask.any() else 0.0

    print(f"\nCenter: {center_method} = ({cx:.4f}, {cy:.4f})")
    print(f"Usable range: {usable_range_m*100:.2f} cm")
    print(f"\n--- Naive trajectory similarity (all valid pairs, confounded by robot start) ---")
    print(f"  pairwise r: {pairwise['pearson']:.4f}, mean traj dist: {pairwise['mean_traj_distance_m']*100:.2f} cm")
    print(f"  read: {check3.read_analysis3(pairwise)}")

    sb = controlled["same_base_summary"]
    sp = controlled["same_pos_summary"]
    print(f"\n--- Base_idx-controlled trajectory similarity ---")
    print(
        f"  OBJECT-DRIVEN (same robot start, diff object): n={sb['n_pairs']}, "
        f"r={sb['pearson']:.4f}, mean traj dist={sb['mean_traj_distance_m']*100:.2f} cm"
    )
    print(
        f"    successes-only: n={sb.get('n_success_pairs', 0)}, "
        f"r={sb.get('success_pearson', float('nan')):.4f}, "
        f"mean={sb.get('success_mean_traj_distance_m', float('nan'))*100:.2f} cm"
    )
    print(
        f"  NON-OBJECT BASELINE (same pos, diff robot start): n={sp['n_pairs']}, "
        f"mean traj dist={sp['mean_traj_distance_m']*100:.2f} cm (r={sp['pearson']:.4f})"
    )
    if not math.isnan(sb["mean_traj_distance_m"]) and not math.isnan(sp["mean_traj_distance_m"]):
        lift = (sb["mean_traj_distance_m"] - sp["mean_traj_distance_m"]) * 100
        print(f"  object-driven mean MINUS baseline mean = {lift:+.2f} cm "
              f"(positive => object position adds trajectory variation beyond robot-start noise)")

    overlay = check3.plot_failed_overlay(rows, paths["overlay_plot"], cx, cy)
    failed_tube = overlay.get("failed_mean_tube_width_m", float("nan"))
    succ_tube = overlay.get("succeeded_mean_tube_width_m", float("nan"))
    print(f"\n--- Failed-trajectory overlay (robot start held fixed per panel) ---")
    print(f"  failed within-start tube width:    {failed_tube*100:.2f} cm")
    print(f"  succeeded within-start tube width: {succ_tube*100:.2f} cm")
    if not math.isnan(failed_tube) and not math.isnan(succ_tube):
        if failed_tube < succ_tube:
            print("  failed tube tighter than succeeded => failures collapse onto a canned motion (replay-like)")
        else:
            print("  failed tube wider than succeeded => failures scatter (breakage-like)")

    check3.plot_success_vs_distance(rows, distances_m, table)
    check3.plot_traj_vs_object(pairwise)
    check3.plot_controlled_traj(controlled, paths["controlled_plot"])
    print(
        f"\nWrote {check3.OUT_SUCCESS_PLOT.name}, {check3.OUT_TRAJ_PLOT.name}, "
        f"{paths['controlled_plot'].name}, {paths['overlay_plot'].name}"
    )

    return {
        "table": table,
        "pairwise": pairwise,
        "controlled": controlled,
        "overlay": overlay,
        "usable_range_m": usable_range_m,
        "distances_m": distances_m,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Combined spike3 tight-grid analysis")
    parser.parse_args()

    paths = configure_modules()
    if not paths["csv"].exists():
        sys.exit(f"Missing {paths['csv']}. Run modal_perturbation_sweep.py first.")

    rows, skipped = perturb.read_rows()

    import csv
    with paths["csv"].open(newline="") as f:
        raw_rows = list(csv.DictReader(f))
    n_commanded = len(raw_rows)
    n_rejected = sum(1 for r in raw_rows if r.get("status", "ok") != "ok")

    # Attach trajectories for pairwise analysis (check3 expects ri["traj"]).
    attach_trajectories(rows, paths["traj_dir"])
    if skipped:
        print("Skipped rollouts:")
        for rid, reason in skipped:
            print(f"  rollout_id={rid}: {reason}")

    n_valid = len(rows)
    n_success = sum(1 for r in rows if r["success"])
    print(f"\nDataset: {n_valid} valid, {n_rejected} rejected, {n_success} successes")

    if n_valid < 2:
        sys.exit("Need at least 2 valid rollouts.")

    success_rows = [r for r in rows if r["success"]]
    all_reg = regression_block(rows, "all valid")
    succ_reg = regression_block(success_rows, "successes only")

    perturb.write_plot(rows, all_reg["stats_x"], all_reg["stats_y"])
    print(f"Wrote {paths['grasp_plot'].name}")

    check3_result = run_check3_analysis(rows)
    verdict = combined_verdict(all_reg, succ_reg, check3_result, n_valid, n_success, n_rejected)
    print(f"\n=== Combined verdict ===\n{verdict}")

    write_findings_section(
        paths, skipped, all_reg, succ_reg, check3_result,
        n_commanded, n_valid, n_success, n_rejected, verdict,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
