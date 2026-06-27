#!/usr/bin/env python3
"""
Score OpenVLA and Pi0.5 on the validated replay-vs-grounding scale.

Run after data collection:
  ./.venv/bin/python compare_models.py
  ./.venv/bin/python compare_models.py --task-id 2
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

import analyze_perturbation as perturb
import analyze_spike3 as spike3
import numpy as np

from libero_task_config import TaskPreset, get_task_preset

ROOT = Path(__file__).resolve().parent

# Task 0 baseline numbers for replication verdict (from COMPARISON.md).
TASK0_BASELINE = {
    "success_gap": 0.403,
    "grounding_delta_matched": 0.183,
    "lift_pi05": 4.41,
    "lift_openvla": 0.09,
    "slope_x_pi05": 0.826,
    "slope_x_openvla": 0.552,
}


def cell_key(row: dict[str, Any]) -> tuple[float, float]:
    """Commanded grid cell (shared identically across models)."""
    return (round(float(row["grid_x"]), 4), round(float(row["grid_y"]), 4))


def grounding_from_rows(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Grasp-slope grounding score from a row subset (successes expected)."""
    reg = spike3._regression_silent(rows)
    sx = reg["stats_x"].get("slope", float("nan"))
    sy = reg["stats_y"].get("slope", float("nan"))
    gain = (abs(sx) + abs(sy)) / 2.0 if not (math.isnan(sx) or math.isnan(sy)) else float("nan")
    grounding = float(np.clip(gain, 0.0, 1.0)) if not math.isnan(gain) else float("nan")
    return {
        "n": len(rows),
        "slope_x": sx,
        "slope_y": sy,
        "gain": gain,
        "grounding": grounding,
        "replay": 1.0 - grounding if not math.isnan(grounding) else float("nan"),
    }


def load_dataset(name: str, csv_path: Path, traj_dir: Path) -> dict[str, Any] | None:
    if not csv_path.is_file():
        print(f"  {name}: missing {csv_path.name}")
        return None
    perturb.configure(csv_path, traj_dir, ROOT / "_noop_grasp.png", label=name)
    rows, skipped = perturb.read_rows()
    spike3.attach_trajectories(rows, traj_dir)
    metrics = spike3.extract_detector_metrics(rows, quiet=True)
    succ_rows = [r for r in rows if r["success"]]
    succ = grounding_from_rows(succ_rows)
    all_gain = metrics["tracking_gain"]
    return {
        "name": name,
        "rows": rows,
        "succ_rows": succ_rows,
        "n_valid": metrics["n_valid"],
        "n_success": metrics["n_success"],
        "success_rate": metrics["n_success"] / metrics["n_valid"] if metrics["n_valid"] else float("nan"),
        "slope_x_all": metrics["slope_x"],
        "slope_y_all": metrics["slope_y"],
        "grounding_all": float(np.clip(all_gain, 0.0, 1.0)) if not math.isnan(all_gain) else float("nan"),
        "replay_all": 1.0 - float(np.clip(all_gain, 0.0, 1.0)) if not math.isnan(all_gain) else float("nan"),
        "slope_x_succ": succ["slope_x"],
        "slope_y_succ": succ["slope_y"],
        "grounding_succ": succ["grounding"],
        "replay_succ": succ["replay"],
        "controlled_lift_cm": metrics["controlled_lift_cm"],
        "controlled_r": metrics["controlled_r"],
        "failed_tube_cm": metrics["failed_tube_cm"],
        "succ_tube_cm": metrics["succ_tube_cm"],
        "skipped": len(skipped),
    }


def add_matched_band(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Recompute grounding on a matched object-position band."""
    succ_cells_per_model = [{cell_key(r) for r in res["succ_rows"]} for res in results]
    shared_cells = set.intersection(*succ_cells_per_model) if succ_cells_per_model else set()

    most_restrictive = min(results, key=lambda res: len({cell_key(r) for r in res["succ_rows"]}))
    bx = [r["obj_x"] for r in most_restrictive["succ_rows"]]
    by = [r["obj_y"] for r in most_restrictive["succ_rows"]]
    bbox = (min(bx), max(bx), min(by), max(by)) if bx else (0.0, 0.0, 0.0, 0.0)

    for res in results:
        intersect_rows = [r for r in res["succ_rows"] if cell_key(r) in shared_cells]
        bbox_rows = [
            r
            for r in res["succ_rows"]
            if bbox[0] <= r["obj_x"] <= bbox[1] and bbox[2] <= r["obj_y"] <= bbox[3]
        ]
        res["matched_intersect"] = grounding_from_rows(intersect_rows)
        res["matched_bbox"] = grounding_from_rows(bbox_rows)

    return {
        "shared_cells": sorted(shared_cells),
        "n_shared_cells": len(shared_cells),
        "bbox": bbox,
        "bbox_ref": most_restrictive["name"],
    }


def replication_verdict(
    openvla: dict[str, Any] | None,
    pi05: dict[str, Any] | None,
) -> tuple[str, str]:
    if not openvla or not pi05:
        return "insufficient data", "Cannot compare — missing model results."

    d_succ = pi05["success_rate"] - openvla["success_rate"]
    d_int = pi05["matched_intersect"]["grounding"] - openvla["matched_intersect"]["grounding"]

    success_replicates = d_succ > 0
    grounding_replicates = (
        not math.isnan(d_int)
        and d_int > 0
        and pi05["matched_intersect"]["n"] > 0
        and openvla["matched_intersect"]["n"] > 0
    )

    if success_replicates and grounding_replicates:
        verdict = "REPLICATES"
        detail = (
            "Same direction on both success rate and matched-band grounding "
            "(Pi0.5 > OpenVLA)."
        )
    elif success_replicates and not grounding_replicates:
        verdict = "PARTIAL"
        detail = (
            "Success gap holds but grounding delta did not replicate "
            "(check n_success for grounding)."
        )
    elif not success_replicates and not grounding_replicates:
        verdict = "DOES NOT REPLICATE"
        detail = "Opposite or null gap — task_0 result may have been task-specific."
    else:
        verdict = "MIXED"
        detail = "Grounding gap without success gap — inspect per-metric."

    return verdict, detail


def write_comparison(
    preset: TaskPreset,
    results: list[dict[str, Any]],
    band: dict[str, Any],
    *,
    replication: bool = False,
) -> None:
    rows_md = "\n".join(
        f"| {r['name']} | {r['n_valid']} | {r['n_success']} | {r['success_rate']:.1%} | "
        f"{r['grounding_succ']:.3f} | {r['replay_succ']:.3f} | "
        f"{r['slope_x_succ']:.3f} | {r['slope_y_succ']:.3f} | "
        f"{r['controlled_lift_cm']:.2f} | {r['failed_tube_cm']:.2f} | {r['succ_tube_cm']:.2f} |"
        for r in results
    )

    matched_md = "\n".join(
        f"| {r['name']} | {r['matched_intersect']['n']} | "
        f"{r['matched_intersect']['grounding']:.3f} | {r['matched_intersect']['slope_x']:.3f} | "
        f"{r['matched_intersect']['slope_y']:.3f} | {r['matched_bbox']['n']} | "
        f"{r['matched_bbox']['grounding']:.3f} |"
        for r in results
    )

    openvla = next((r for r in results if "OpenVLA" in r["name"]), None)
    pi05 = next((r for r in results if "Pi0.5" in r["name"]), None)

    if openvla and pi05:
        d_raw = pi05["grounding_succ"] - openvla["grounding_succ"]
        d_int = pi05["matched_intersect"]["grounding"] - openvla["matched_intersect"]["grounding"]
        d_box = pi05["matched_bbox"]["grounding"] - openvla["matched_bbox"]["grounding"]
        d_succ = pi05["success_rate"] - openvla["success_rate"]
        read = (
            f"- **Success-rate gap (confound-free):** Pi0.5 {pi05['success_rate']:.1%} vs "
            f"OpenVLA {openvla['success_rate']:.1%} = {d_succ:+.1%} on the identical valid set.\n"
            f"- **Grounding delta (raw successes-only, different subsets):** {d_raw:+.3f} "
            f"(Pi0.5 {pi05['grounding_succ']:.3f} vs OpenVLA {openvla['grounding_succ']:.3f}).\n"
            f"- **Grounding delta (matched shared-cell band):** {d_int:+.3f} "
            f"(Pi0.5 {pi05['matched_intersect']['grounding']:.3f} vs "
            f"OpenVLA {openvla['matched_intersect']['grounding']:.3f}) over "
            f"{band['n_shared_cells']} identical cells — **this is the defensible number**.\n"
            f"- **Grounding delta (matched bbox, robustness):** {d_box:+.3f} "
            f"(Pi0.5 {pi05['matched_bbox']['grounding']:.3f} vs "
            f"OpenVLA {openvla['matched_bbox']['grounding']:.3f})."
        )
    elif openvla:
        read = "Pi0.5 data not yet collected; OpenVLA baseline shown below."
    else:
        read = "Insufficient data for comparison."

    replication_section = ""
    summary_para = ""
    if replication and openvla and pi05:
        verdict, detail = replication_verdict(openvla, pi05)
        d_succ = pi05["success_rate"] - openvla["success_rate"]
        d_int = pi05["matched_intersect"]["grounding"] - openvla["matched_intersect"]["grounding"]
        b = TASK0_BASELINE

        def rep_cell(task_val: float, baseline: float, higher_is_pi05: bool = True) -> str:
            if math.isnan(task_val):
                return "n/a"
            same = (task_val > 0 if higher_is_pi05 else task_val < 0) if baseline != 0 else task_val == baseline
            if baseline > 0 and task_val > 0:
                same = True
            elif baseline > 0 and task_val <= 0:
                same = False
            return "yes" if same else "no"

        replication_section = f"""
## Replication verdict (vs task_id=0)

| metric | task_0 | task_{preset.task_id} | replicates? |
|--------|--------|--------|-------------|
| success gap | +{b['success_gap']:.1%} | {d_succ:+.1%} | {rep_cell(d_succ, b['success_gap'])} |
| grounding delta (matched) | +{b['grounding_delta_matched']:.3f} | {d_int:+.3f} | {rep_cell(d_int, b['grounding_delta_matched'])} |
| lift Pi0.5 / OpenVLA | {b['lift_pi05']:.2f} / {b['lift_openvla']:.2f} | {pi05['controlled_lift_cm']:.2f} / {openvla['controlled_lift_cm']:.2f} | direction |
| x-slope Pi0.5 / OpenVLA | {b['slope_x_pi05']:.3f} / {b['slope_x_openvla']:.3f} | {pi05['matched_intersect']['slope_x']:.3f} / {openvla['matched_intersect']['slope_x']:.3f} | direction |

**Verdict: {verdict}** — {detail}

OpenVLA n_success={openvla['n_success']}, Pi0.5 n_success={pi05['n_success']} (task_0: 16 / 41).
"""
        summary_para = (
            f"On task_id={preset.task_id} ({preset.instruction}), the Pi0.5 vs OpenVLA gap "
            f"{'**replicates**' if verdict == 'REPLICATES' else '**does not fully replicate**'} "
            f"task_0: success gap {d_succ:+.1%} (task_0 +40.3%), matched grounding delta "
            f"{d_int:+.3f} (task_0 +0.183). This is replication across two LIBERO-Spatial tasks, "
            f"not a claim of generalization."
        )

    title_suffix = f" (task_id={preset.task_id})" if preset.task_id != 0 else ""
    content = f"""# OpenVLA vs Pi0.5 — Replay vs Grounding Comparison{title_suffix}

**Task:** `{preset.instruction}`

Same perturbation grid: 5×5 over ±7 cm centered at ({preset.center_x:.3f}, {preset.center_y:.3f}), 3 repeats per cell (75 commanded).
Override rejections are model-independent (check fires before policy runs).

## Summary table (raw successes-only grounding score)

| model | n_valid | n_success | success_rate | grounding (succ) | replay (succ) | slope_x | slope_y | lift (cm) | failed tube | succ tube |
|-------|---------|-----------|--------------|------------------|---------------|---------|---------|-----------|-------------|-----------|
{rows_md}

## Matched object-position band (removes successes-on-different-subsets confound)

- **Shared-cell intersection:** the {band['n_shared_cells']} grid cells where BOTH models have ≥1 success.
- **Bounding box ({band['bbox_ref']} successes):** x∈[{band['bbox'][0]:.3f}, {band['bbox'][1]:.3f}],
  y∈[{band['bbox'][2]:.3f}, {band['bbox'][3]:.3f}].

| model | n (shared cells) | grounding (matched) | slope_x | slope_y | n (bbox) | grounding (bbox) |
|-------|------------------|---------------------|---------|---------|----------|------------------|
{matched_md}

## Read
{read}
{replication_section}
## Scale definition
- `grounding_score = clip(mean(|slope_x|, |slope_y|), 0, 1)` on successes-only grasp regression
- `replay_score = 1 - grounding_score`
- Anchored by synthetic controls (see VALIDATION.md / controls_task2/)

## Notes
- Lead with the **success-rate gap** — pure counting on the identical valid set.
- **Matched shared-cell grounding delta** is the defensible grounding headline.
- Two tasks = replicated finding, not generalization claim.
"""
    if summary_para:
        content += f"\n## Plain-English summary\n\n{summary_para}\n"

    preset.comparison_md.write_text(content)
    print(f"Wrote {preset.comparison_md}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare OpenVLA vs Pi0.5 grounding scores")
    parser.add_argument("--task-id", type=int, default=0, help="LIBERO spatial task id")
    args = parser.parse_args()
    preset = get_task_preset(args.task_id)

    datasets = {
        f"OpenVLA (task {preset.task_id})": {
            "csv": preset.openvla_csv,
            "traj_dir": preset.openvla_traj_dir,
        },
        "Pi0.5": {
            "csv": preset.pi05_csv,
            "traj_dir": preset.pi05_traj_dir,
        },
    }

    print(f"=== Model comparison (task_id={preset.task_id}) ===")
    results: list[dict[str, Any]] = []
    for label, paths in datasets.items():
        print(f"\n{label}:")
        scored = load_dataset(label, paths["csv"], paths["traj_dir"])
        if scored:
            print(
                f"  n={scored['n_valid']}, successes={scored['n_success']} ({scored['success_rate']:.1%}), "
                f"grounding(succ)={scored['grounding_succ']:.3f}, replay(succ)={scored['replay_succ']:.3f}"
            )
            results.append(scored)

    if not results:
        print("No datasets found.", file=sys.stderr)
        return 1

    band = add_matched_band(results)
    print(
        f"\nMatched band: {band['n_shared_cells']} shared success cells; "
        f"bbox ({band['bbox_ref']}) x∈[{band['bbox'][0]:.3f},{band['bbox'][1]:.3f}] "
        f"y∈[{band['bbox'][2]:.3f},{band['bbox'][3]:.3f}]"
    )
    for res in results:
        mi = res["matched_intersect"]
        mb = res["matched_bbox"]
        print(
            f"  {res['name']}: matched-cell grounding={mi['grounding']:.3f} (n={mi['n']}, "
            f"sx={mi['slope_x']:.3f}, sy={mi['slope_y']:.3f}); bbox grounding={mb['grounding']:.3f} (n={mb['n']})"
        )

    write_comparison(preset, results, band, replication=(preset.task_id != 0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
