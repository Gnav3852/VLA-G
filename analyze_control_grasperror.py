#!/usr/bin/env python3
"""
Deciding diagnostic: does grasp-error-vs-displacement detect a KNOWN replayer?

Compares synthetic controls (replayer g=0, oracle g=1 from make_controls.py) against
real OpenVLA / Pi0.5 rollouts at:
  - narrow range: ±7 cm grid (existing position sweep)
  - wide range:   ±20 cm synthetic grid (same fixed grasp / ideal oracle)

Run:
  ./.venv/bin/python analyze_control_grasperror.py
  ./.venv/bin/python analyze_control_grasperror.py --task-id 2
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

from analyze_position_grasperror import (
    bootstrap_slope_ci,
    linear_regression,
    load_rollouts,
)
from libero_task_config import TaskPreset, get_task_preset

ROOT = Path(__file__).resolve().parent
OUT_JSON = ROOT / "control_grasperror_diagnostic.json"
OUT_PLOT = ROOT / "grasp_error_vs_displacement_controls.png"
OUT_MD = ROOT / "CONTROL_DIAGNOSTIC.md"

NARROW_HALF_M = 0.07
WIDE_HALF_M = 0.20
WIDE_STEP_M = 0.025
DEFAULT_SOURCE_ROLLOUT_ID = 6

# Detection thresholds (pre-registered for this diagnostic)
REPLAYER_R2_WORKS = 0.30
REPLAYER_SLOPE_MIN = 0.20
ORACLE_SLOPE_MAX = 0.15


def parse_bool(value: str) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def load_control_csv(csv_path: Path, center_x: float, center_y: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with csv_path.open(newline="") as f:
        for raw in csv.DictReader(f):
            if raw.get("status", "ok") != "ok":
                continue
            try:
                ox = float(raw["actual_object_init_x"])
                oy = float(raw["actual_object_init_y"])
                gx = float(raw["grasp_point_x"])
                gy = float(raw["grasp_point_y"])
            except (KeyError, TypeError, ValueError):
                continue
            rows.append(
                {
                    "perturbation_distance_m": float(math.hypot(ox - center_x, oy - center_y)),
                    "grasp_error_m": float(math.hypot(gx - ox, gy - oy)),
                    "obj_x": ox,
                    "obj_y": oy,
                    "grasp_x": gx,
                    "grasp_y": gy,
                    "success": parse_bool(raw.get("success", "")),
                }
            )
    return rows


def find_successful_source_rollout(preset: TaskPreset) -> int:
    with preset.openvla_csv.open(newline="") as f:
        rows = [
            r
            for r in csv.DictReader(f)
            if r.get("status", "ok") == "ok" and parse_bool(r.get("success", ""))
        ]
    if not rows:
        raise RuntimeError(f"No successful rollouts in {preset.openvla_csv}")
    rows.sort(key=lambda r: abs(float(r["grid_x"])) + abs(float(r["grid_y"])))
    return int(rows[0]["rollout_id"])


def load_source_grasp(preset: TaskPreset, source_rollout_id: int | None) -> tuple[np.ndarray, np.ndarray, int]:
    rid = source_rollout_id
    if rid is None:
        rid = DEFAULT_SOURCE_ROLLOUT_ID if preset.task_id == 0 else find_successful_source_rollout(preset)
    traj_path = preset.openvla_traj_dir / f"rollout_{rid:03d}.json"
    traj = np.asarray(json.loads(traj_path.read_text()), dtype=np.float64)
    k = int(np.where(traj[:, 2] == traj[:, 2].min())[0][0])
    gp0 = traj[k, :2].copy()
    with preset.openvla_csv.open(newline="") as f:
        for raw in csv.DictReader(f):
            if int(raw["rollout_id"]) == rid:
                o0 = np.array(
                    [float(raw["actual_object_init_x"]), float(raw["actual_object_init_y"])],
                    dtype=np.float64,
                )
                return gp0, o0, rid
    raise RuntimeError(f"source rollout {rid} missing from {preset.openvla_csv}")


def build_wide_grid(center_x: float, center_y: float, half_m: float, step_m: float) -> list[tuple[float, float]]:
    xs = np.arange(center_x - half_m, center_x + half_m + 1e-9, step_m)
    ys = np.arange(center_y - half_m, center_y + half_m + 1e-9, step_m)
    return [(float(x), float(y)) for x in xs for y in ys]


def synthesize_wide_controls(
    center_x: float,
    center_y: float,
    fixed_grasp_xy: np.ndarray,
    *,
    oracle_noise_m: float = 0.0,
    rng: np.random.Generator | None = None,
) -> dict[str, list[dict[str, Any]]]:
    rng = rng or np.random.default_rng(42)
    replayer: list[dict[str, Any]] = []
    oracle: list[dict[str, Any]] = []
    for ox, oy in build_wide_grid(center_x, center_y, WIDE_HALF_M, WIDE_STEP_M):
        pd = float(math.hypot(ox - center_x, oy - center_y))
        gx_r, gy_r = float(fixed_grasp_xy[0]), float(fixed_grasp_xy[1])
        ge_r = float(math.hypot(gx_r - ox, gy_r - oy))
        replayer.append(
            {
                "perturbation_distance_m": pd,
                "grasp_error_m": ge_r,
                "obj_x": ox,
                "obj_y": oy,
                "grasp_x": gx_r,
                "grasp_y": gy_r,
            }
        )
        if oracle_noise_m > 0:
            gx_o = ox + float(rng.normal(0.0, oracle_noise_m))
            gy_o = oy + float(rng.normal(0.0, oracle_noise_m))
        else:
            gx_o, gy_o = ox, oy
        ge_o = float(math.hypot(gx_o - ox, gy_o - oy))
        oracle.append(
            {
                "perturbation_distance_m": pd,
                "grasp_error_m": ge_o,
                "obj_x": ox,
                "obj_y": oy,
                "grasp_x": gx_o,
                "grasp_y": gy_o,
            }
        )
    return {"replayer": replayer, "oracle": oracle}


def regression_from_points(rows: list[dict[str, Any]], *, label: str) -> dict[str, Any]:
    if not rows:
        return {"label": label, "n": 0, "slope": math.nan, "r2": math.nan, "slope_ci_95": {}}
    x = np.asarray([r["perturbation_distance_m"] for r in rows], dtype=np.float64)
    y = np.asarray([r["grasp_error_m"] for r in rows], dtype=np.float64)
    stats = linear_regression(x, y)
    stats["slope_ci_95"] = bootstrap_slope_ci(x, y)
    stats["label"] = label
    stats["perturbation_span_m"] = float(x.max() - x.min()) if len(x) else math.nan
    stats["mean_grasp_error_m"] = float(np.mean(y))
    stats["max_grasp_error_m"] = float(np.max(y))
    return stats


def load_real_model_successes(preset: TaskPreset) -> dict[str, list[dict[str, Any]]]:
    cx, cy = preset.center_x, preset.center_y
    out: dict[str, list[dict[str, Any]]] = {}
    for model, csv_path, traj_dir in (
        ("openvla", preset.openvla_csv, preset.openvla_traj_dir),
        ("pi05", preset.pi05_csv, preset.pi05_traj_dir),
    ):
        if not csv_path.is_file():
            out[model] = []
            continue
        rows, _ = load_rollouts(csv_path, traj_dir, cx, cy)
        out[model] = [
            {
                "perturbation_distance_m": r["perturbation_distance_m"],
                "grasp_error_m": r["grasp_error_m"],
            }
            for r in rows
            if r["success"]
        ]
    return out


def replayer_detected(reg: dict[str, Any]) -> bool:
    ci = reg.get("slope_ci_95", {})
    lo = ci.get("lo", math.nan)
    return (
        reg.get("slope", math.nan) >= REPLAYER_SLOPE_MIN
        and lo > 0
        and reg.get("r2", math.nan) >= REPLAYER_R2_WORKS
    )


def oracle_flat(reg: dict[str, Any]) -> bool:
    ci = reg.get("slope_ci_95", {})
    return abs(reg.get("slope", math.nan)) <= ORACLE_SLOPE_MAX and ci.get("lo", math.nan) < ORACLE_SLOPE_MAX


def classify_verdict(
    narrow_rep: dict[str, Any],
    wide_rep: dict[str, Any],
    narrow_oracle: dict[str, Any],
    wide_oracle: dict[str, Any],
) -> tuple[str, list[str]]:
    n_works = replayer_detected(narrow_rep)
    w_works = replayer_detected(wide_rep)
    n_oracle = oracle_flat(narrow_oracle)
    w_oracle = oracle_flat(wide_oracle)

    rationale = [
        f"Narrow replayer: slope={narrow_rep['slope']:.3f}, R²={narrow_rep['r2']:.3f}, "
        f"CI=[{narrow_rep['slope_ci_95'].get('lo', math.nan):.3f}, {narrow_rep['slope_ci_95'].get('hi', math.nan):.3f}]",
        f"Wide replayer: slope={wide_rep['slope']:.3f}, R²={wide_rep['r2']:.3f}, "
        f"CI=[{wide_rep['slope_ci_95'].get('lo', math.nan):.3f}, {wide_rep['slope_ci_95'].get('hi', math.nan):.3f}]",
        f"Narrow oracle (make_controls g=1): slope={narrow_oracle['slope']:.3f}, "
        f"mean error={narrow_oracle['mean_grasp_error_m']*100:.1f}cm (flat vs displacement)",
        f"Wide ideal oracle: slope={wide_oracle['slope']:.3f}, mean error={wide_oracle['mean_grasp_error_m']*100:.1f}cm",
    ]

    if n_works and w_works and n_oracle and w_oracle:
        return (
            "METRIC WORKS: replayer shows clean rising error, oracle flat. "
            "Therefore the real models' FLAT result is REAL → OpenVLA and Pi0.5 both ground position; "
            "their score gap is NOT a position-grounding gap. This is a genuine (honest-negative) finding.",
            rationale,
        )

    if w_works and w_oracle and not n_works:
        return (
            "METRIC WORKS ONLY AT WIDE RANGE: replayer detection fails at ±7cm (R² below threshold) "
            "but succeeds at ±20cm. → the real-model range was too narrow; real models must be re-tested "
            "at wider displacement before any conclusion. Detector is fine, experiment range was off.",
            rationale
            + [
                f"At ±7cm: positive slope ({narrow_rep['slope']:.2f}) but R²={narrow_rep['r2']:.2f} "
                f"(below {REPLAYER_R2_WORKS} bar) — not a reliable detection.",
            ],
        )

    if not w_works:
        return (
            "METRIC BROKEN: replayer is ALSO flat even at wide range. → the metric cannot detect replay "
            "even when replay is GUARANTEED. The position probe as built does not work and needs redesign or abandonment.",
            rationale,
        )

    # Partial: wide works, narrow ambiguous slope but low R²
    if w_works and not n_works:
        return (
            "METRIC WORKS ONLY AT WIDE RANGE: replayer detection fails at ±7cm (R² below threshold) "
            "but succeeds at ±20cm. → the real-model range was too narrow; real models must be re-tested "
            "at wider displacement before any conclusion. Detector is fine, experiment range was off.",
            rationale,
        )

    return (
        "METRIC BROKEN: replayer is ALSO flat even at wide range. → the metric cannot detect replay "
        "even when replay is GUARANTEED. The position probe as built does not work and needs redesign or abandonment.",
        rationale,
    )


def plot_overlay(
    series: dict[str, dict[str, list[dict[str, Any]]]],
    regressions: dict[str, dict[str, dict[str, Any]]],
    out_path: Path,
    task_id: int,
) -> None:
    styles = {
        "replayer": {"c": "#d62728", "m": "s", "label": "replayer (g=0)"},
        "oracle": {"c": "#9467bd", "m": "^", "label": "oracle (g=1)"},
        "openvla": {"c": "#1f77b4", "m": "o", "label": "OpenVLA (successes)"},
        "pi05": {"c": "#2ca02c", "m": "D", "label": "Pi0.5 (successes)"},
    }
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
    titles = [
        f"Narrow ±{NARROW_HALF_M*100:.0f}cm (task {task_id})",
        f"Wide ±{WIDE_HALF_M*100:.0f}cm (synthetic)",
    ]
    for ax, range_key, title in zip(axes, ("narrow", "wide"), titles):
        for name, style in styles.items():
            pts = series[range_key].get(name, [])
            if not pts:
                continue
            x = [p["perturbation_distance_m"] * 100 for p in pts]
            y = [p["grasp_error_m"] * 100 for p in pts]
            ax.scatter(x, y, c=style["c"], marker=style["m"], s=22, alpha=0.55, label=style["label"])
            reg = regressions[range_key].get(name, {})
            if reg.get("n", 0) >= 2 and not math.isnan(reg.get("slope", math.nan)):
                xs = np.linspace(min(x), max(x), 50)
                ys = reg["slope"] * (xs / 100) + reg["intercept"] * 100
                ax.plot(xs, ys, color=style["c"], lw=1.2, ls="--", alpha=0.85)
        ax.set_xlabel("Perturbation distance from center (cm)")
        ax.set_title(title)
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel("Grasp error (cm)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.02), fontsize=8)
    fig.suptitle("Grasp-error vs displacement: controls vs real models", y=1.06, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def analyze_task(task_id: int, source_rollout_id: int | None) -> dict[str, Any]:
    preset = get_task_preset(task_id)
    cx, cy = preset.center_x, preset.center_y
    controls_dir = preset.controls_dir

    replayer_csv = controls_dir / "ctrl_g0p00_clean" / "rollouts.csv"
    oracle_csv = controls_dir / "ctrl_g1p00_clean" / "rollouts.csv"
    if not replayer_csv.is_file() or not oracle_csv.is_file():
        raise FileNotFoundError(
            f"Missing controls under {controls_dir}. "
            f"Run: ./.venv/bin/python make_controls.py --task-id {task_id}"
        )

    narrow_replayer_pts = load_control_csv(replayer_csv, cx, cy)
    narrow_oracle_pts = load_control_csv(oracle_csv, cx, cy)
    real = load_real_model_successes(preset)

    gp0, _o0, rid = load_source_grasp(preset, source_rollout_id)
    wide = synthesize_wide_controls(cx, cy, gp0)

    series = {
        "narrow": {
            "replayer": narrow_replayer_pts,
            "oracle": narrow_oracle_pts,
            "openvla": real.get("openvla", []),
            "pi05": real.get("pi05", []),
        },
        "wide": {
            "replayer": wide["replayer"],
            "oracle": wide["oracle"],
        },
    }

    regressions: dict[str, dict[str, dict[str, Any]]] = {"narrow": {}, "wide": {}}
    results: dict[str, Any] = {
        "task_id": task_id,
        "center_x": cx,
        "center_y": cy,
        "source_rollout_id": rid,
    }

    mapping = [
        ("narrow", "replayer", narrow_replayer_pts, "narrow_replayer"),
        ("narrow", "oracle", narrow_oracle_pts, "narrow_oracle"),
        ("narrow", "openvla", real.get("openvla", []), "narrow_openvla"),
        ("narrow", "pi05", real.get("pi05", []), "narrow_pi05"),
        ("wide", "replayer", wide["replayer"], "wide_replayer"),
        ("wide", "oracle", wide["oracle"], "wide_oracle"),
    ]
    for range_key, name, pts, result_key in mapping:
        reg = regression_from_points(pts, label=f"{range_key}_{name}")
        regressions[range_key][name] = reg
        results[result_key] = reg

    verdict, rationale = classify_verdict(
        results["narrow_replayer"],
        results["wide_replayer"],
        results["narrow_oracle"],
        results["wide_oracle"],
    )
    results["verdict"] = verdict
    results["rationale"] = rationale
    results["series"] = series
    results["regressions"] = regressions
    return results


def write_md_all(task_results: list[dict[str, Any]], path: Path) -> None:
    primary = task_results[0]
    lines = [
        "# Control Diagnostic: Grasp-Error vs Displacement",
        "",
        "Synthetic replayer (g=0, fixed trajectory) and oracle (g=1) from `make_controls.py`, "
        "compared to real OpenVLA / Pi0.5 successes.",
        "",
        "## Detection thresholds",
        "",
        f"- Replayer **works**: slope ≥ {REPLAYER_SLOPE_MIN}, 95% CI excludes 0, R² ≥ {REPLAYER_R2_WORKS}",
        f"- Oracle **flat**: |slope| ≤ {ORACLE_SLOPE_MAX}",
        "",
        "## Primary verdict (task 0)",
        "",
        f"**{primary['verdict']}**",
        "",
    ]
    lines.extend(primary["rationale"])

    for tr in task_results:
        tid = tr["task_id"]
        lines.extend(
            [
                "",
                f"## Task {tid} (source rollout {tr['source_rollout_id']})",
                "",
                "### Narrow ±7 cm",
                "",
                "| Policy | n | slope | R² | 95% CI | mean error |",
                "|--------|---|-------|-----|--------|------------|",
            ]
        )
        for key, label in (
            ("narrow_replayer", "Replayer g=0"),
            ("narrow_oracle", "Oracle g=1"),
            ("narrow_openvla", "OpenVLA"),
            ("narrow_pi05", "Pi0.5"),
        ):
            r = tr[key]
            ci = r.get("slope_ci_95", {})
            lines.append(
                f"| {label} | {r['n']} | {r['slope']:.3f} | {r['r2']:.3f} | "
                f"[{ci.get('lo', math.nan):.3f}, {ci.get('hi', math.nan):.3f}] | "
                f"{r.get('mean_grasp_error_m', math.nan)*100:.1f} cm |"
            )
        lines.extend(
            [
                "",
                "### Wide ±20 cm (synthetic)",
                "",
                "| Policy | n | slope | R² | 95% CI | mean error |",
                "|--------|---|-------|-----|--------|------------|",
            ]
        )
        for key, label in (("wide_replayer", "Replayer"), ("wide_oracle", "Oracle (ideal)")):
            r = tr[key]
            ci = r.get("slope_ci_95", {})
            lines.append(
                f"| {label} | {r['n']} | {r['slope']:.3f} | {r['r2']:.3f} | "
                f"[{ci.get('lo', math.nan):.3f}, {ci.get('hi', math.nan):.3f}] | "
                f"{r.get('mean_grasp_error_m', math.nan)*100:.1f} cm |"
            )
        if tid != primary["task_id"]:
            lines.extend(["", f"Task {tid} verdict: **{tr['verdict']}**"])

    lines.extend(
        [
            "",
            "## Implications",
            "",
            "Real models at ±7 cm show flat grasp-error vs displacement (R²<0.02). "
            "This diagnostic determines whether that flatness is a metric artifact or a real finding.",
            "",
            "- **If metric works (or works at wide range):** flat real models may genuinely ground position.",
            "- **If metric broken:** real-model flatness tells us nothing; probe needs redesign.",
            "",
            "## Caveats",
            "- Narrow oracle = make_controls trajectory shift (constant offset, flat slope).",
            "- Wide oracle = idealized grasp-at-object (not min-z EE).",
            "- Wide replayer uses fixed grasp from source rollout min-z point.",
        ]
    )
    path.write_text("\n".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser(description="Control diagnostic for grasp-error metric")
    parser.add_argument("--task-ids", type=int, nargs="*", default=[0, 2])
    parser.add_argument("--source-rollout-id", type=int, default=None)
    args = parser.parse_args()

    task_results = [analyze_task(tid, args.source_rollout_id if tid == 0 else None) for tid in args.task_ids]
    primary = task_results[0]

    plot_overlay(primary["series"], primary["regressions"], OUT_PLOT, primary["task_id"])
    write_md_all(task_results, OUT_MD)

    def _json_default(obj: Any) -> Any:
        if isinstance(obj, (np.floating, float)) and math.isnan(obj):
            return None
        if isinstance(obj, np.integer):
            return int(obj)
        raise TypeError(type(obj))

    payload = {"tasks": [{k: v for k, v in tr.items() if k not in ("series", "regressions")} for tr in task_results]}
    OUT_JSON.write_text(json.dumps(payload, indent=2, default=_json_default))

    print(f"Wrote {OUT_JSON}")
    print(f"Wrote {OUT_PLOT}")
    print(f"Wrote {OUT_MD}")
    print(f"\nPrimary verdict (task 0): {primary['verdict']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
