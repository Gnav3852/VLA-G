"""Load VLA-Trace paper numbers and compute derived white-box scalars."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
NUMBERS_PATH = ROOT / "vla_trace_numbers.json"

SUITES = ("libero_10", "libero_object", "libero_spatial", "libero_goal")


def load_numbers(path: Path | None = None) -> dict[str, Any]:
    with (path or NUMBERS_PATH).open() as f:
        return json.load(f)


def _baseline_map(patchmask_rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for row in patchmask_rows:
        if row["setting"] != "baseline":
            continue
        out[row["model"]] = {suite: float(row[suite]) for suite in SUITES}
    return out


def patchmask_conditions(numbers: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Per (model, setting, suite) white-box PatchMask condition rows."""
    numbers = numbers or load_numbers()
    baselines = _baseline_map(numbers["table4_patchmask"])
    rows: list[dict[str, Any]] = []
    for entry in numbers["table4_patchmask"]:
        if entry["setting"] == "baseline":
            continue
        model = entry["model"]
        for suite in SUITES:
            masked_sr = float(entry[suite])
            baseline_sr = baselines[model][suite]
            delta_sr = baseline_sr - masked_sr
            rel_drop = delta_sr / baseline_sr if baseline_sr > 0 else float("nan")
            rows.append(
                {
                    "model": model,
                    "setting": entry["setting"],
                    "suite": suite,
                    "baseline_sr": baseline_sr,
                    "masked_sr": masked_sr,
                    "delta_sr": delta_sr,
                    "relative_drop": rel_drop,
                    "avg_drop_paper": entry.get("avg_drop"),
                }
            )
    return rows


def knockout_visual_dependency(numbers: dict[str, Any] | None = None) -> dict[str, float]:
    """(baseline - gen_no_image) / baseline averaged across suites (Table 2)."""
    numbers = numbers or load_numbers()
    out: dict[str, float] = {}
    for model in ("pi05", "openvla"):
        baseline = next(
            r
            for r in numbers["table2_knockout_all_layer"]
            if r["model"] == model and r["prefill"] and r["gen_text"] and r["gen_image"]
        )
        no_image = next(
            r
            for r in numbers["table2_knockout_all_layer"]
            if r["model"] == model and r["prefill"] and r["gen_text"] and not r["gen_image"]
        )
        deps = []
        for suite in SUITES:
            b = float(baseline[suite])
            n = float(no_image[suite])
            if b > 0:
                deps.append((b - n) / b)
        out[model] = sum(deps) / len(deps) if deps else float("nan")
    return out


def attention_mass(numbers: dict[str, Any] | None = None) -> dict[str, float]:
    numbers = numbers or load_numbers()
    out: dict[str, float] = {}
    for row in numbers["table3_attention_iou"]:
        if row["stage"] == "full":
            out[row["model"]] = float(row["mass_object"])
    return out


def derived_per_model(numbers: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    numbers = numbers or load_numbers()
    visual_dep = knockout_visual_dependency(numbers)
    attn = attention_mass(numbers)
    avg_target_bg_drop: dict[str, float] = {}
    for row in numbers["table4_patchmask"]:
        if row["setting"] == "target_bg" and row["avg_drop"] is not None:
            avg_target_bg_drop[row["model"]] = float(row["avg_drop"])

    models = sorted(set(list(visual_dep) + list(avg_target_bg_drop)))
    rows = []
    for model in models:
        rows.append(
            {
                "model": model,
                "visual_dependency": visual_dep.get(model, float("nan")),
                "attention_mass_object": attn.get(model, float("nan")),
                "patchmask_target_bg_avg_drop": avg_target_bg_drop.get(model, float("nan")),
            }
        )
    return rows
