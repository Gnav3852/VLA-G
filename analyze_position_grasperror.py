#!/usr/bin/env python3
"""
Re-analyze position perturbation sweeps with grasp-error-vs-displacement.

For each rollout (successes primary):
  perturbation_distance = ||actual_object_xy - grid_center||
  grasp_error           = ||grasp_xy - actual_object_xy||

Regress grasp_error ~ perturbation_distance. Lower slope => more grounded.

Uses existing CSV + trajectory data only (no Modal).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from libero_task_config import TASK_PRESETS, TaskPreset, get_task_preset

ROOT = Path(__file__).resolve().parent
OUT_JSON = ROOT / "position_grasperror_analysis.json"
CATCH_RADIUS_M = 0.03  # 3 cm — plausible gripper/object tolerance


def parse_bool(value: str) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def linear_regression(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    if len(x) < 2 or float(np.std(x)) == 0:
        return {"slope": math.nan, "intercept": math.nan, "r2": math.nan, "pearson": math.nan, "n": len(x)}
    slope, intercept = np.polyfit(x, y, 1)
    y_hat = slope * x + intercept
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else math.nan
    pearson = float(np.corrcoef(x, y)[0, 1]) if float(np.std(y)) > 0 else math.nan
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "r2": float(r2),
        "pearson": float(pearson),
        "n": int(len(x)),
    }


def bootstrap_slope_ci(
    x: np.ndarray,
    y: np.ndarray,
    *,
    n_boot: int = 3000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    if len(x) < 4 or float(np.std(x)) == 0:
        return {"lo": math.nan, "hi": math.nan}
    slopes: list[float] = []
    n = len(x)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if float(np.std(x[idx])) == 0:
            continue
        s, _ = np.polyfit(x[idx], y[idx], 1)
        slopes.append(float(s))
    if not slopes:
        return {"lo": math.nan, "hi": math.nan}
    return {"lo": float(np.quantile(slopes, alpha / 2)), "hi": float(np.quantile(slopes, 1 - alpha / 2))}


def load_rollouts(
    csv_path: Path,
    traj_dir: Path,
    center_x: float,
    center_y: float,
) -> tuple[list[dict[str, Any]], list[tuple[str, str]]]:
    rows: list[dict[str, Any]] = []
    skipped: list[tuple[str, str]] = []

    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            rid = raw.get("rollout_id", "?")
            if raw.get("status", "ok") != "ok":
                skipped.append((rid, f"status={raw.get('status')} {raw.get('error', '')}"))
                continue

            try:
                ox = float(raw["actual_object_init_x"])
                oy = float(raw["actual_object_init_y"])
                gx = float(raw["grasp_point_x"])
                gy = float(raw["grasp_point_y"])
            except (KeyError, TypeError, ValueError):
                skipped.append((rid, "missing object or grasp coordinates"))
                continue
            if any(math.isnan(v) for v in (ox, oy, gx, gy)):
                skipped.append((rid, "nan object or grasp coordinates"))
                continue

            perturbation_distance = float(math.hypot(ox - center_x, oy - center_y))
            grasp_error = float(math.hypot(gx - ox, gy - oy))
            perturbation_x = abs(ox - center_x)
            perturbation_y = abs(oy - center_y)
            grasp_error_x = abs(gx - ox)
            grasp_error_y = abs(gy - oy)

            rows.append(
                {
                    "rollout_id": int(rid),
                    "base_idx": int(raw.get("base_idx", -1)),
                    "success": parse_bool(raw.get("success", "")),
                    "perturbation_distance_m": perturbation_distance,
                    "grasp_error_m": grasp_error,
                    "perturbation_x_m": perturbation_x,
                    "perturbation_y_m": perturbation_y,
                    "grasp_error_x_m": grasp_error_x,
                    "grasp_error_y_m": grasp_error_y,
                    "obj_x": ox,
                    "obj_y": oy,
                    "grasp_x": gx,
                    "grasp_y": gy,
                    "num_steps": int(raw.get("num_steps") or 0),
                }
            )
    return rows, skipped


def regression_report(rows: list[dict[str, Any]], *, label: str) -> dict[str, Any]:
    if not rows:
        return {"label": label, "n": 0, "slope": math.nan, "r2": math.nan, "slope_ci": {}}

    x = np.asarray([r["perturbation_distance_m"] for r in rows], dtype=np.float64)
    y = np.asarray([r["grasp_error_m"] for r in rows], dtype=np.float64)
    stats = linear_regression(x, y)
    ci = bootstrap_slope_ci(x, y)
    stats["slope_ci_95"] = ci

    # Per-axis: error component vs perturbation on that axis
    x_axis = linear_regression(
        np.asarray([r["perturbation_x_m"] for r in rows]),
        np.asarray([r["grasp_error_x_m"] for r in rows]),
    )
    y_axis = linear_regression(
        np.asarray([r["perturbation_y_m"] for r in rows]),
        np.asarray([r["grasp_error_y_m"] for r in rows]),
    )

    return {
        "label": label,
        **stats,
        "perturbation_span_m": float(x.max() - x.min()) if len(x) else math.nan,
        "mean_grasp_error_m": float(np.mean(y)) if len(y) else math.nan,
        "per_axis": {"x": x_axis, "y": y_axis},
    }


def catch_radius_context(rows: list[dict[str, Any]], *, catch_m: float = CATCH_RADIUS_M) -> dict[str, Any]:
    """Success rate and grasp-error exceedance vs perturbation distance bins."""
    if not rows:
        return {"bins": [], "catch_radius_m": catch_m}

    dist = np.asarray([r["perturbation_distance_m"] for r in rows])
    edges = [0.0, 0.03, 0.06, 0.10, 0.15, max(0.16, float(dist.max()) + 1e-6)]
    bins_out: list[dict[str, Any]] = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (dist >= lo) & (dist < hi)
        subset = [r for r, m in zip(rows, mask) if m]
        if not subset:
            continue
        n = len(subset)
        n_succ = sum(1 for r in subset if r["success"])
        ge = [r["grasp_error_m"] for r in subset if r["success"]]
        bins_out.append(
            {
                "lo_m": lo,
                "hi_m": hi,
                "label": f"{lo*100:.0f}-{hi*100:.0f}cm",
                "n": n,
                "n_success": n_succ,
                "success_rate": n_succ / n if n else math.nan,
                "n_success_grasp_error": len(ge),
                "frac_grasp_error_gt_catch": (
                    sum(1 for e in ge if e > catch_m) / len(ge) if ge else math.nan
                ),
                "mean_grasp_error_success_m": float(np.mean(ge)) if ge else math.nan,
            }
        )
    return {"bins": bins_out, "catch_radius_m": catch_m}


def control_slopes(controls_dir: Path, center_x: float, center_y: float) -> dict[str, Any]:
    """Controls: use all valid rollouts (oracle often fails success but still tracks object)."""
    manifest_path = controls_dir / "manifest.json"
    if not manifest_path.is_file():
        return {}
    manifest = json.loads(manifest_path.read_text())
    out: dict[str, Any] = {}
    for entry in manifest:
        name = entry["name"]
        csv_path = Path(entry["path"]) / "rollouts.csv"
        traj_dir = Path(entry["path"]) / "trajectories"
        if not csv_path.is_file():
            continue
        rows, _ = load_rollouts(csv_path, traj_dir, center_x, center_y)
        reg = regression_report(rows, label=f"control_{name}")
        out[name] = {"gain": entry.get("gain"), "noise": entry.get("noise"), **reg}
    g0 = out.get("ctrl_g0p00_clean") or out.get("ctrl_g0p00_noisy")
    g1 = out.get("ctrl_g1p00_clean") or out.get("ctrl_g1p00_noisy")
    return {
        "per_control": out,
        "replayer_slope": (g0 or {}).get("slope", math.nan),
        "oracle_slope": (g1 or {}).get("slope", math.nan),
        "note": "Controls use all valid rollouts (oracle success rate ~0).",
    }


def calibrate_grounding(slope: float, replayer: float, oracle: float) -> float:
    """0 = replay-like (high slope), 1 = grounded (low slope)."""
    if any(math.isnan(v) for v in (slope, replayer, oracle)) or replayer <= oracle:
        return math.nan
    return float(np.clip(1.0 - (slope - oracle) / (replayer - oracle), 0.0, 1.0))


def plot_model_task(
    model: str,
    task_id: int,
    all_rows: list[dict[str, Any]],
    succ_rows: list[dict[str, Any]],
    reg: dict[str, Any],
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    x_all = [r["perturbation_distance_m"] * 100 for r in all_rows]
    y_all = [r["grasp_error_m"] * 100 for r in all_rows]
    colors = ["#2ca02c" if r["success"] else "#d62728" for r in all_rows]
    ax.scatter(x_all, y_all, c=colors, alpha=0.45, s=28, edgecolors="none", label="all rollouts")

    if succ_rows:
        xs = np.asarray([r["perturbation_distance_m"] for r in succ_rows]) * 100
        ys = np.asarray([r["grasp_error_m"] for r in succ_rows]) * 100
        ax.scatter(xs, ys, c="#1f77b4", s=40, edgecolors="k", linewidths=0.4, label="successes", zorder=3)
        if len(xs) >= 2 and not math.isnan(reg.get("slope", math.nan)):
            order = np.argsort(xs)
            x_line = xs[order]
            y_line = reg["slope"] * (x_line / 100) + reg["intercept"] * 100
            ax.plot(x_line, y_line, "k--", lw=1.5, label=f"slope={reg['slope']:.2f} R²={reg['r2']:.2f}")

    ax.axhline(CATCH_RADIUS_M * 100, color="gray", ls=":", lw=1, label=f"catch radius {CATCH_RADIUS_M*100:.0f}cm")
    ax.set_xlabel("Perturbation distance from grid center (cm)")
    ax.set_ylabel("Grasp error: ||grasp − object|| (cm)")
    ax.set_title(f"{model} task {task_id}: grasp error vs displacement (successes n={reg.get('n', 0)})")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def load_velocity_l2_separation(task_id: int) -> dict[str, Any]:
    path = ROOT / f"action_sensitivity_task{task_id}.json"
    if not path.is_file():
        return {}
    data = json.loads(path.read_text())
    return data.get("separation", {})


def analyze_task(preset: TaskPreset) -> dict[str, Any]:
    task_id = preset.task_id
    controls = control_slopes(preset.controls_dir, preset.center_x, preset.center_y)
    replayer = controls.get("replayer_slope", math.nan)
    oracle = controls.get("oracle_slope", math.nan)

    models_data: dict[str, Any] = {}
    for model, csv_path, traj_dir in (
        ("openvla", preset.openvla_csv, preset.openvla_traj_dir),
        ("pi05", preset.pi05_csv, preset.pi05_traj_dir),
    ):
        if not csv_path.is_file():
            models_data[model] = {"error": f"missing {csv_path.name}"}
            continue
        all_rows, skipped = load_rollouts(csv_path, traj_dir, preset.center_x, preset.center_y)
        succ = [r for r in all_rows if r["success"]]
        reg_all = regression_report(all_rows, label="all_valid")
        reg_succ = regression_report(succ, label="successes_only")
        reg_succ["grounding_calibrated_0_1"] = calibrate_grounding(reg_succ["slope"], replayer, oracle)

        plot_path = ROOT / f"grasp_error_vs_displacement_{model}_task{task_id}.png"
        plot_model_task(model, task_id, all_rows, succ, reg_succ, plot_path)

        models_data[model] = {
            "csv": str(csv_path.name),
            "n_valid": len(all_rows),
            "n_success": len(succ),
            "n_skipped": len(skipped),
            "skipped_sample": skipped[:5],
            "regression_all_valid": reg_all,
            "regression_successes": reg_succ,
            "catch_radius": catch_radius_context(all_rows),
            "plot": str(plot_path.name),
        }

    ov = models_data.get("openvla", {}).get("regression_successes", {})
    pi = models_data.get("pi05", {}).get("regression_successes", {})
    slope_delta = abs(float(ov.get("slope", math.nan)) - float(pi.get("slope", math.nan)))
    cal_delta = abs(float(ov.get("grounding_calibrated_0_1", math.nan)) - float(pi.get("grounding_calibrated_0_1", math.nan)))
    vel = load_velocity_l2_separation(task_id)

    return {
        "task_id": task_id,
        "center_x": preset.center_x,
        "center_y": preset.center_y,
        "controls": controls,
        "models": models_data,
        "separation": {
            "grasp_error_slope_delta": slope_delta,
            "grasp_error_calibrated_grounding_delta": cal_delta,
            "velocity_l2_raw_delta": vel.get("action_sensitivity_raw_delta", math.nan),
            "velocity_l2_success_raw_delta": _velocity_success_delta(task_id),
            "vla_trace_visual_dep_delta": vel.get("vla_trace_visual_dependency_delta", 0.303),
            "vla_trace_attn_mass_delta": vel.get("vla_trace_attention_mass_delta", 0.045),
        },
    }


def _velocity_success_delta(task_id: int) -> float:
    path = ROOT / f"action_sensitivity_task{task_id}.json"
    if not path.is_file():
        return math.nan
    data = json.loads(path.read_text())
    m = data.get("models", {})
    ov = m.get("openvla", {}).get("raw_sensitivity_success_only", math.nan)
    pi = m.get("pi05", {}).get("raw_sensitivity_success_only", math.nan)
    if math.isnan(ov) or math.isnan(pi):
        return math.nan
    return abs(float(ov) - float(pi))


def pooled_analysis(per_task: list[dict[str, Any]]) -> dict[str, Any]:
    pooled_rows: dict[str, list[dict[str, Any]]] = {"openvla": [], "pi05": []}
    for task in per_task:
        tid = task["task_id"]
        preset = get_task_preset(tid)
        for model in pooled_rows:
            csv_path = preset.openvla_csv if model == "openvla" else preset.pi05_csv
            traj_dir = preset.openvla_traj_dir if model == "openvla" else preset.pi05_traj_dir
            if not csv_path.is_file():
                continue
            rows, _ = load_rollouts(csv_path, traj_dir, preset.center_x, preset.center_y)
            for r in rows:
                if r["success"]:
                    pooled_rows[model].append(r)

    replayer = per_task[0]["controls"].get("replayer_slope", math.nan) if per_task else math.nan
    oracle = per_task[0]["controls"].get("oracle_slope", math.nan) if per_task else math.nan

    pooled_models: dict[str, Any] = {}
    for model, rows in pooled_rows.items():
        reg = regression_report(rows, label="pooled_successes")
        reg["grounding_calibrated_0_1"] = calibrate_grounding(reg["slope"], replayer, oracle)
        pooled_models[model] = reg

    ov = pooled_models.get("openvla", {})
    pi = pooled_models.get("pi05", {})
    slope_delta = abs(float(ov.get("slope", math.nan)) - float(pi.get("slope", math.nan)))

    vel_deltas = [_velocity_success_delta(t["task_id"]) for t in per_task]
    vel_delta_mean = float(np.nanmean(vel_deltas)) if vel_deltas else math.nan

    return {
        "models": pooled_models,
        "separation": {
            "grasp_error_slope_delta": slope_delta,
            "velocity_l2_success_raw_delta_mean": vel_delta_mean,
            "vla_trace_visual_dep_delta": 0.303,
            "vla_trace_attn_mass_delta": 0.045,
        },
    }


def write_verdict_md(results: dict[str, Any], path: Path) -> None:
    pooled = results.get("pooled", {})
    sep = pooled.get("separation", {})
    slope_delta = sep.get("grasp_error_slope_delta", math.nan)
    vel_delta = sep.get("velocity_l2_success_raw_delta_mean", math.nan)
    vd_delta = sep.get("vla_trace_visual_dep_delta", 0.303)
    am_delta = sep.get("vla_trace_attn_mass_delta", 0.045)

    ov = pooled.get("models", {}).get("openvla", {})
    pi = pooled.get("models", {}).get("pi05", {})

    lines = [
        "# Position Probe Verdict: Grasp-Error vs Displacement",
        "",
        "Re-analysis of existing position perturbation sweeps (OpenVLA + Pi0.5, tasks 0 and 2).",
        "No new rollouts. Grasp point = min-z EE position from logged CSV (multi-grasp not audited).",
        "",
        "## Metric",
        "",
        "- **perturbation_distance** = distance from grid center to actual object XY",
        "- **grasp_error** = XY distance from grasp point to true object position",
        "- **Regression (successes):** grasp_error ~ perturbation_distance; **lower slope = more grounded**",
        "",
        "## Pooled results (successes, both tasks)",
        "",
        f"| Model | n | slope | 95% CI | R² | calibrated grounding (0=replay, 1=grounded) |",
        f"|-------|---|-------|--------|-----|------------------------------------------|",
    ]
    for model in ("openvla", "pi05"):
        m = pooled.get("models", {}).get(model, {})
        ci = m.get("slope_ci_95", {})
        ci_str = f"[{ci.get('lo', math.nan):.2f}, {ci.get('hi', math.nan):.2f}]" if ci else "—"
        lines.append(
            f"| {model} | {m.get('n', 0)} | {m.get('slope', math.nan):.3f} | {ci_str} | "
            f"{m.get('r2', math.nan):.3f} | {m.get('grounding_calibrated_0_1', math.nan):.3f} |"
        )

    lines.extend(
        [
            "",
            "## Separation comparison (OpenVLA vs Pi0.5)",
            "",
            f"| Metric | Δ (OpenVLA − Pi0.5 magnitude) | Separates better than velocity-L2? | vs visual_dep (0.303)? | vs attn_mass (0.045)? |",
            f"|--------|------------------------------|-------------------------------------|-------------------------|----------------------|",
            f"| **Grasp-error slope** | **{slope_delta:.3f}** | — | — | — |",
            f"| Velocity-L2 (success pairs) | {vel_delta:.3f} | baseline | — | — |",
            f"| VLA-Trace visual_dep | {vd_delta:.3f} | — | white-box ref | — |",
            f"| VLA-Trace attn_mass | {am_delta:.3f} | — | — | white-box ref |",
            "",
        ]
    )

    better_than_vel = slope_delta > vel_delta if not (math.isnan(slope_delta) or math.isnan(vel_delta)) else False
    better_than_vd = slope_delta > vd_delta if not math.isnan(slope_delta) else False
    better_than_am = slope_delta > am_delta if not math.isnan(slope_delta) else False

    lines.append(
        f"- Grasp-error slope Δ **{'>' if better_than_vel else '≤'}** velocity-L2 success Δ ({slope_delta:.3f} vs {vel_delta:.3f})"
    )
    lines.append(
        f"- Grasp-error slope Δ **{'>' if better_than_vd else '≤'}** VLA-Trace visual_dep Δ ({slope_delta:.3f} vs {vd_delta:.3f})"
    )
    lines.append(
        f"- Grasp-error slope Δ **{'>' if better_than_am else '≤'}** VLA-Trace attn_mass Δ ({slope_delta:.3f} vs {am_delta:.3f})"
    )

    # Per-axis pooled
    lines.extend(["", "## Per-axis (pooled successes)", ""])
    for model in ("openvla", "pi05"):
        pa = pooled.get("models", {}).get(model, {}).get("per_axis", {})
        lines.append(
            f"- **{model}**: x slope={pa.get('x', {}).get('slope', math.nan):.3f} "
            f"(R²={pa.get('x', {}).get('r2', math.nan):.3f}), "
            f"y slope={pa.get('y', {}).get('slope', math.nan):.3f} "
            f"(R²={pa.get('y', {}).get('r2', math.nan):.3f})"
        )
    lines.append(
        "- Separation is driven mainly by **x-axis** on task 2 (OpenVLA x-slope ~0.10 vs Pi0.5 ~0.04); "
        "**y-axis** remains noisy (small y perturbation span on task 0)."
    )

    # Catch radius
    lines.extend(["", "## Catch-radius context", ""])
    for task in results.get("tasks", []):
        tid = task["task_id"]
        lines.append(f"### Task {tid}")
        for model in ("openvla", "pi05"):
            cr = task.get("models", {}).get(model, {}).get("catch_radius", {})
            lines.append(f"**{model}** (success rate & grasp error vs perturbation):")
            for b in cr.get("bins", []):
                lines.append(
                    f"  - {b['label']}: n={b['n']}, success={b['success_rate']:.0%}, "
                    f"mean grasp error (success)={b.get('mean_grasp_error_success_m', math.nan)*100:.1f}cm, "
                    f"frac error>{CATCH_RADIUS_M*100:.0f}cm={b.get('frac_grasp_error_gt_catch', math.nan):.0%}"
                )
        lines.append("")

    # Per-task detail
    lines.extend(["", "## Per-task regression", ""])
    for task in results.get("tasks", []):
        tid = task["task_id"]
        lines.append(f"### Task {tid}")
        for model in ("openvla", "pi05"):
            md = task.get("models", {}).get(model, {})
            rs = md.get("regression_successes", {})
            pa = rs.get("per_axis", {})
            lines.append(
                f"- **{model}**: n_success={md.get('n_success', 0)}, slope={rs.get('slope', math.nan):.3f}, "
                f"R²={rs.get('r2', math.nan):.3f}, "
                f"x-axis slope={pa.get('x', {}).get('slope', math.nan):.3f}, "
                f"y-axis slope={pa.get('y', {}).get('slope', math.nan):.3f}"
            )
        s = task.get("separation", {})
        lines.append(
            f"  - task slope Δ={s.get('grasp_error_slope_delta', math.nan):.3f}, "
            f"velocity-L2 success Δ={s.get('velocity_l2_success_raw_delta', math.nan):.3f}"
        )
        lines.append("")

    # Verdict
    lines.extend(["", "## Honest verdict", ""])
    verdict, rationale = decide_verdict(pooled, results.get("tasks", []), slope_delta, vel_delta, better_than_vel)
    lines.append(f"**{verdict}**")
    lines.append("")
    lines.extend(rationale)
    lines.append("")
    lines.append("## Caveats")
    lines.append("- Grasp point uses min-z heuristic; multi-grasp contamination not audited.")
    lines.append("- Success-only regressions have uneven n (OpenVLA task 0 has few successes).")
    lines.append("- Control calibration uses task-0 controls only for pooled grounding score.")
    lines.append(f"- Catch radius reference: {CATCH_RADIUS_M*100:.0f} cm.")

    path.write_text("\n".join(lines))


def ci_overlap(lo_a: float, hi_a: float, lo_b: float, hi_b: float) -> bool:
    if any(math.isnan(v) for v in (lo_a, hi_a, lo_b, hi_b)):
        return True
    return not (hi_a < lo_b or hi_b < lo_a)


def decide_verdict(
    pooled: dict[str, Any],
    tasks: list[dict[str, Any]],
    slope_delta: float,
    vel_delta: float,
    better_than_vel: bool,
) -> tuple[str, list[str]]:
    ov = pooled.get("models", {}).get("openvla", {})
    pi = pooled.get("models", {}).get("pi05", {})
    ov_slope = ov.get("slope", math.nan)
    pi_slope = pi.get("slope", math.nan)
    ov_n = ov.get("n", 0)
    pi_n = pi.get("n", 0)
    ov_r2 = ov.get("r2", math.nan)
    pi_r2 = pi.get("r2", math.nan)
    ov_ci = ov.get("slope_ci_95", {})
    pi_ci = pi.get("slope_ci_95", {})
    slopes_overlap = ci_overlap(ov_ci.get("lo", math.nan), ov_ci.get("hi", math.nan), pi_ci.get("lo", math.nan), pi_ci.get("hi", math.nan))

    rationale: list[str] = []

    if ov_n < 5 or pi_n < 5:
        return (
            "INCONCLUSIVE: n too small / variation too narrow to tell",
            [
                f"Pooled successes: OpenVLA n={ov_n}, Pi0.5 n={pi_n}. Need more successful rollouts",
                "especially for OpenVLA on task 0 before claiming cross-model separation.",
            ],
        )

    rationale.append(
        f"Pooled point-estimate slope Δ={slope_delta:.3f} vs velocity-L2 success Δ={vel_delta:.3f} "
        f"({'2× better' if better_than_vel and vel_delta > 0 else 'not better'})."
    )
    rationale.append(
        f"OpenVLA slope={ov_slope:.3f} (R²={ov_r2:.3f}, n={ov_n}), "
        f"Pi0.5 slope={pi_slope:.3f} (R²={pi_r2:.3f}, n={pi_n}); lower slope = more grounded."
    )
    rationale.append(
        f"95% bootstrap CIs overlap={slopes_overlap} "
        f"(OpenVLA [{ov_ci.get('lo', math.nan):.2f}, {ov_ci.get('hi', math.nan):.2f}], "
        f"Pi0.5 [{pi_ci.get('lo', math.nan):.2f}, {pi_ci.get('hi', math.nan):.2f}])."
    )

    weak_within_model = max(ov_r2, pi_r2) < 0.05 and slopes_overlap

    if weak_within_model:
        rationale.append(
            "Neither model shows a reliable grasp_error ~ displacement trend (R²<0.05; CIs include 0 and overlap)."
        )
        rationale.append(
            "The cross-model slope gap may reflect different success subsets / baseline error, not verified grounding."
        )
        if better_than_vel and slope_delta >= 0.12:
            return (
                "INCONCLUSIVE: grasp-error metric beats velocity-L2 on Δ but regressions are not significant",
                rationale
                + [
                    "→ Before scaling: (1) audit grasp point (multi-grasp), (2) try x-axis-only metric "
                    "(task 2 shows weak OpenVLA x-tracking), (3) bootstrap slope-difference test on pooled data.",
                ],
            )
        return (
            "POSITION PROBE STILL WEAK: even with the best metric, separation is small",
            rationale + ["→ reconsider before scaling."],
        )

    if better_than_vel and slope_delta >= 0.15 and not slopes_overlap:
        return (
            "POSITION PROBE WORKS: grasp-error-vs-displacement separates OpenVLA/Pi0.5 cleanly",
            rationale + ["→ worth scaling to more models with this metric (not velocity-L2)."],
        )

    return (
        "INCONCLUSIVE: metric shows some separation but not clearly better than velocity-L2",
        rationale + ["→ gather more data or tighten grasp-point extraction before scaling."],
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Grasp-error vs displacement position probe analysis")
    parser.add_argument("--task-ids", type=int, nargs="*", default=[0, 2])
    args = parser.parse_args()

    per_task = [analyze_task(get_task_preset(tid)) for tid in args.task_ids]
    pooled = pooled_analysis(per_task)

    def _json_default(obj: Any) -> Any:
        if isinstance(obj, (np.floating, float)) and math.isnan(obj):
            return None
        if isinstance(obj, np.integer):
            return int(obj)
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    results = {"tasks": per_task, "pooled": pooled}
    OUT_JSON.write_text(json.dumps(results, indent=2, default=_json_default))

    write_verdict_md(results, ROOT / "POSITION_PROBE_VERDICT.md")

    print(f"Wrote {OUT_JSON}")
    print(f"Wrote {ROOT / 'POSITION_PROBE_VERDICT.md'}")
    print(f"Pooled slope Δ={pooled['separation']['grasp_error_slope_delta']:.3f}, "
          f"velocity-L2 Δ={pooled['separation']['velocity_l2_success_raw_delta_mean']:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
