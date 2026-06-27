#!/usr/bin/env python3
"""
Validate replay-vs-grounding detector on synthetic gain-sweep controls, then score OpenVLA.

Run:
  ./.venv/bin/python make_controls.py
  ./.venv/bin/python validate_detector.py
  ./.venv/bin/python validate_detector.py --task-id 2
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import analyze_check3 as check3
import analyze_perturbation as perturb
import analyze_spike3 as spike3
import matplotlib.pyplot as plt
import numpy as np

from libero_task_config import get_task_preset

ROOT = Path(__file__).resolve().parent


def load_control_rows(csv_path: Path, traj_dir: Path) -> list[dict[str, Any]]:
    perturb.configure(csv_path, traj_dir, ROOT / "_noop_grasp.png", label="control")
    rows, _ = perturb.read_rows()
    spike3.attach_trajectories(rows, traj_dir)
    return rows


def recovered_gain(metrics: dict[str, Any]) -> float:
    return float(metrics["tracking_gain"])


def grounding_score(metrics: dict[str, Any]) -> float:
    """0 = full replay, 1 = full grounding (anchored at controls)."""
    g = recovered_gain(metrics)
    if math.isnan(g):
        return float("nan")
    return float(np.clip(g, 0.0, 1.0))


def replay_score(metrics: dict[str, Any]) -> float:
    gs = grounding_score(metrics)
    if math.isnan(gs):
        return float("nan")
    return 1.0 - gs


def evaluate_pass_fail(results: list[dict[str, Any]]) -> dict[str, Any]:
    clean = [r for r in results if r["noise"] == "clean"]
    true_g = np.array([r["true_gain"] for r in clean])
    rec_g = np.array([r["recovered_gain"] for r in clean])

    if len(clean) >= 2 and np.std(true_g) > 0:
        cal_slope, cal_intercept = np.polyfit(true_g, rec_g, 1)
        y_hat = cal_slope * true_g + cal_intercept
        ss_res = float(np.sum((rec_g - y_hat) ** 2))
        ss_tot = float(np.sum((rec_g - np.mean(rec_g)) ** 2))
        cal_r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    else:
        cal_slope = cal_intercept = cal_r2 = float("nan")

    g0 = next((r for r in clean if r["true_gain"] == 0.0), None)
    g1 = next((r for r in clean if r["true_gain"] == 1.0), None)

    replayer_ok = (
        g0 is not None
        and abs(g0["recovered_gain"]) < 0.1
        and (math.isnan(g0["controlled_lift_cm"]) or g0["controlled_lift_cm"] < 0.5)
    )
    oracle_ok = (
        g1 is not None
        and 0.85 <= g1["recovered_gain"] <= 1.15
    )
    sweep_ok = (not math.isnan(cal_r2)) and cal_r2 > 0.95 and (not math.isnan(cal_slope)) and 0.85 <= cal_slope <= 1.15

    passed = replayer_ok and oracle_ok and sweep_ok
    return {
        "passed": passed,
        "replayer_ok": replayer_ok,
        "oracle_ok": oracle_ok,
        "sweep_ok": sweep_ok,
        "cal_slope": cal_slope,
        "cal_intercept": cal_intercept,
        "cal_r2": cal_r2,
        "g0_recovered": g0["recovered_gain"] if g0 else float("nan"),
        "g1_recovered": g1["recovered_gain"] if g1 else float("nan"),
    }


def plot_calibration(results: list[dict[str, Any]], pf: dict[str, Any], calibration_plot: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    for noise, marker, alpha in (("clean", "o", 1.0), ("noisy", "s", 0.6)):
        subset = [r for r in results if r["noise"] == noise]
        tg = [r["true_gain"] for r in subset]
        rg = [r["recovered_gain"] for r in subset]
        axes[0].scatter(tg, rg, marker=marker, s=70, alpha=alpha, label=noise)
    lim = (-0.05, 1.05)
    axes[0].plot(lim, lim, "k--", linewidth=1, label="y=x")
    if not math.isnan(pf["cal_slope"]):
        xx = np.linspace(0, 1, 50)
        axes[0].plot(xx, pf["cal_slope"] * xx + pf["cal_intercept"], "r-", linewidth=1.2,
                     label=f"fit R²={pf['cal_r2']:.3f}")
    axes[0].set_xlim(lim)
    axes[0].set_ylim(lim)
    axes[0].set_xlabel("True tracking gain g")
    axes[0].set_ylabel("Recovered gain (mean |slope| x,y)")
    axes[0].set_title("Grasp regression calibration")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    clean = [r for r in results if r["noise"] == "clean"]
    axes[1].plot([r["true_gain"] for r in clean], [r["controlled_lift_cm"] for r in clean], "o-", color="#4C72B0")
    axes[1].set_xlabel("True gain g")
    axes[1].set_ylabel("Controlled lift (cm)")
    axes[1].set_title("Object-driven lift vs gain")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot([r["true_gain"] for r in clean], [r["controlled_r"] for r in clean], "o-", color="#55A868")
    axes[2].set_xlabel("True gain g")
    axes[2].set_ylabel("Controlled traj-object r")
    axes[2].set_title("Trajectory correlation vs gain")
    axes[2].grid(True, alpha=0.3)

    status = "PASS" if pf["passed"] else "FAIL"
    fig.suptitle(f"Detector validation ({status})", fontsize=12)
    fig.tight_layout()
    fig.savefig(calibration_plot, dpi=150)
    plt.close(fig)


def write_validation_md(
    results: list[dict[str, Any]],
    pf: dict[str, Any],
    openvla: dict[str, Any],
    *,
    validation_md: Path,
    calibration_plot: Path,
    task_label: str,
) -> None:
    clean_rows = "\n".join(
        f"| {r['true_gain']:.2f} | {r['recovered_gain']:.4f} | {r['slope_x']:.4f} | {r['slope_y']:.4f} | "
        f"{r['controlled_lift_cm']:.2f} | {r['controlled_r']:.4f} | {r['n_success']} |"
        for r in results
        if r["noise"] == "clean"
    )
    noisy_rows = "\n".join(
        f"| {r['true_gain']:.2f} | {r['recovered_gain']:.4f} | {r['controlled_lift_cm']:.2f} | {r['controlled_r']:.4f} |"
        for r in results
        if r["noise"] == "noisy"
    )

    ov = openvla
    gs_all = grounding_score(ov)
    rs_all = replay_score(ov)
    gs_succ = float("nan")
    rs_succ = float("nan")
    if not math.isnan(ov.get("succ_tracking_gain", float("nan"))):
        gs_succ = float(np.clip(ov["succ_tracking_gain"], 0.0, 1.0))
        rs_succ = 1.0 - gs_succ

    if pf["passed"]:
        verdict = (
            f"**PASS** — Detector recovers synthetic gain across the sweep "
            f"(calibration R²={pf['cal_r2']:.4f}, slope={pf['cal_slope']:.3f}). "
            f"Replayer g=0 recovered {pf['g0_recovered']:.3f}; oracle g=1 recovered {pf['g1_recovered']:.3f}."
        )
        openvla_read = (
            f"OpenVLA spike3 (successes only, n=16): grounding score = **{gs_succ:.3f}** "
            f"(replay score = {rs_succ:.3f}) — partial/compressed grounding on a validated 0–1 scale.\n"
            f"- All-valid (n=62): grounding = {gs_all:.3f}, replay = {rs_all:.3f} (diluted by non-tracking failures)."
        )
    else:
        verdict = (
            f"**FAIL** — Detector does not cleanly separate synthetic extremes or recover gain. "
            f"Replayer ok={pf['replayer_ok']}, oracle ok={pf['oracle_ok']}, sweep ok={pf['sweep_ok']} "
            f"(R²={pf['cal_r2']:.3f}, slope={pf['cal_slope']:.3f}). "
            "Do not proceed to Pi0.5; consider approach-phase metrics or stronger trajectory discriminators."
        )
        openvla_read = (
            f"OpenVLA raw tracking gain = {ov['tracking_gain']:.3f} — **uncalibrated** until controls pass."
        )

    content = f"""# Detector Validation (synthetic gain sweep)

## Goal
Validate that grasp-vs-object regression (mean |slope| on x,y) recovers known tracking gain
from synthetic controls before scoring real models.

## Control design
- **Positive control (g=0):** recorded successful spike3 trajectory (rollout 6) replayed verbatim; only object position varies.
- **Oracle (g=1):** same backbone with grasp displaced 1:1 toward each object (ramped xy shift to grasp step).
- **Sweep:** g ∈ {{0, 0.25, 0.5, 0.75, 1.0}} × {{clean, noisy (σ=0.5 cm/step)}} over the 62 valid spike3 object positions.

## Pass/fail criteria
- g=0: recovered |gain| < 0.1, lift < 0.5 cm
- g=1: recovered gain ∈ [0.85, 1.15]
- Sweep (clean): linear fit recovered vs true gain has R² > 0.95 and slope ∈ [0.85, 1.15]

## Result: {verdict}

## Calibration table (clean)

| true g | recovered | slope_x | slope_y | lift (cm) | traj r | n_success |
|--------|-----------|---------|---------|-----------|--------|-----------|
{clean_rows}

## Calibration table (noisy)

| true g | recovered | lift (cm) | traj r |
|--------|-----------|-----------|--------|
{noisy_rows}

## Replay-score definition
- `grounding_score = clip(mean(|slope_x|, |slope_y|), 0, 1)` — 0 = replay, 1 = oracle
- `replay_score = 1 - grounding_score`

## OpenVLA placement (spike3)
**Primary (successes only, n=16):**
- slope_x = {ov.get('succ_slope_x', float('nan')):.4f}, slope_y = {ov.get('succ_slope_y', float('nan')):.4f}
- tracking_gain = {ov.get('succ_tracking_gain', float('nan')):.4f}
- {openvla_read}

**All valid (n=62):** slope_x = {ov['slope_x']:.4f}, slope_y = {ov['slope_y']:.4f}, tracking_gain = {ov['tracking_gain']:.4f}

## Honesty notes
- Successes-only controlled traj-r is partly tautological; **grasp slope is the primary calibration anchor**.
- Synthetic controls validate metric math, not all real-world failure modes.
- Noisy controls inject per-step xy noise only; they do not simulate physics or vision.

Plot: `{calibration_plot.name}`
"""
    validation_md.write_text(content)
    print(f"Wrote {validation_md}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate detector on synthetic controls")
    parser.add_argument("--task-id", type=int, default=0)
    args = parser.parse_args()
    preset = get_task_preset(args.task_id)
    controls_dir = preset.controls_dir
    manifest_path = controls_dir / "manifest.json"
    calibration_plot = ROOT / (
        "control_calibration.png" if args.task_id == 0 else f"control_calibration_task{args.task_id}.png"
    )
    validation_md = ROOT / ("VALIDATION.md" if args.task_id == 0 else f"VALIDATION_task{args.task_id}.md")
    model_csv = preset.openvla_csv
    model_traj = preset.openvla_traj_dir
    task_label = f"task_id={args.task_id}"

    if not manifest_path.is_file():
        print(f"Run make_controls.py --task-id {args.task_id} first.", file=sys.stderr)
        return 1

    manifest = json.loads(manifest_path.read_text())
    results: list[dict[str, Any]] = []

    print("=== Control datasets ===")
    for entry in manifest:
        out_dir = Path(entry["path"])
        csv_path = out_dir / "rollouts.csv"
        traj_dir = out_dir / "trajectories"
        rows = load_control_rows(csv_path, traj_dir)
        metrics = spike3.extract_detector_metrics(rows, quiet=True)
        rec = recovered_gain(metrics)
        results.append(
            {
                "name": entry["name"],
                "true_gain": float(entry["gain"]),
                "noise": entry["noise"],
                "recovered_gain": rec,
                "slope_x": metrics["slope_x"],
                "slope_y": metrics["slope_y"],
                "controlled_lift_cm": metrics["controlled_lift_cm"],
                "controlled_r": metrics["controlled_r"],
                "failed_tube_cm": metrics["failed_tube_cm"],
                "succ_tube_cm": metrics["succ_tube_cm"],
                "n_success": metrics["n_success"],
                "n_valid": metrics["n_valid"],
            }
        )
        print(
            f"  {entry['name']}: true g={entry['gain']:.2f} -> recovered={rec:.4f} "
            f"(sx={metrics['slope_x']:.3f}, sy={metrics['slope_y']:.3f}, "
            f"lift={metrics['controlled_lift_cm']:.2f}cm, n_succ={metrics['n_success']})"
        )

    pf = evaluate_pass_fail(results)
    print(f"\n=== Pass/fail ===")
    print(f"  replayer (g=0): {'OK' if pf['replayer_ok'] else 'FAIL'} (recovered={pf['g0_recovered']:.4f})")
    print(f"  oracle (g=1):   {'OK' if pf['oracle_ok'] else 'FAIL'} (recovered={pf['g1_recovered']:.4f})")
    print(f"  sweep:          {'OK' if pf['sweep_ok'] else 'FAIL'} (R²={pf['cal_r2']:.4f}, slope={pf['cal_slope']:.4f})")
    print(f"  OVERALL:        {'PASS' if pf['passed'] else 'FAIL'}")

    plot_calibration(results, pf, calibration_plot)
    print(f"Wrote {calibration_plot}")

    print(f"\n=== OpenVLA ({task_label}) ===")
    if not model_csv.is_file():
        print("  model data missing; skipping OpenVLA score")
        openvla = {
            "tracking_gain": float("nan"),
            "slope_x": float("nan"),
            "slope_y": float("nan"),
            "succ_tracking_gain": float("nan"),
            "succ_slope_x": float("nan"),
            "succ_slope_y": float("nan"),
        }
    else:
        perturb.configure(model_csv, model_traj, ROOT / "_noop_grasp.png", label="openvla")
        ov_rows, _ = perturb.read_rows()
        spike3.attach_trajectories(ov_rows, model_traj)
        openvla = spike3.extract_detector_metrics(ov_rows, quiet=True)
        succ_rows = [r for r in ov_rows if r["success"]]
        succ_reg = spike3._regression_silent(succ_rows)
        sx = succ_reg["stats_x"].get("slope", float("nan"))
        sy = succ_reg["stats_y"].get("slope", float("nan"))
        openvla["succ_slope_x"] = sx
        openvla["succ_slope_y"] = sy
        openvla["succ_tracking_gain"] = (
            (abs(sx) + abs(sy)) / 2.0 if not (math.isnan(sx) or math.isnan(sy)) else float("nan")
        )
        gs_all = grounding_score(openvla)
        gs_succ = float(np.clip(openvla["succ_tracking_gain"], 0.0, 1.0))
        print(
            f"  all-valid tracking_gain={openvla['tracking_gain']:.4f} "
            f"(grounding={gs_all:.4f}, replay={1-gs_all:.4f})"
        )
        print(
            f"  successes-only tracking_gain={openvla['succ_tracking_gain']:.4f} "
            f"(grounding={gs_succ:.4f}, replay={1-gs_succ:.4f})"
        )

    write_validation_md(
        results, pf, openvla,
        validation_md=validation_md,
        calibration_plot=calibration_plot,
        task_label=task_label,
    )
    return 0 if pf["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
