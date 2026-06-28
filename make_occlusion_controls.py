#!/usr/bin/env python3
"""
Synthetic occlusion controls: replayer (masked==unmasked) vs oracle (large shift).

Extends the position-perturbation control framework for Probe C calibration.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import patchmask_probe as probe

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "controls_occlusion"


def synth_action_streams(kind: str, n: int = 50, dim: int = 7) -> tuple[list[np.ndarray], list[np.ndarray]]:
    rng = np.random.default_rng(0)
    base = rng.normal(0, 0.1, size=(n, dim))
    unmasked = [base[i].copy() for i in range(n)]
    if kind == "replayer":
        masked = [base[i].copy() for i in range(n)]
    else:
        rng2 = np.random.default_rng(99)
        masked_arr = base + rng2.normal(0, 0.8, size=(n, dim))
        masked = [masked_arr[i] for i in range(n)]
    return unmasked, masked


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate synthetic occlusion controls")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    replayer_unmasked, replayer_masked = synth_action_streams("replayer")
    oracle_unmasked, oracle_masked = synth_action_streams("oracle")

    low_raw = probe.action_stream_l2_shift(replayer_unmasked, replayer_masked)
    high_raw = probe.action_stream_l2_shift(oracle_unmasked, oracle_masked)

    cases = {
        "replayer": (replayer_unmasked, replayer_masked),
        "oracle": (oracle_unmasked, oracle_masked),
    }
    records = []
    for name, (u, m) in cases.items():
        metrics = probe.occlusion_sensitivity_from_streams(u, m, anchor_low=low_raw, anchor_high=high_raw)
        records.append({"name": name, **metrics})

    manifest = {
        "anchors": {"low_raw_l2": low_raw, "high_raw_l2": high_raw},
        "cases": records,
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))
    print(f"Wrote {args.out_dir / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
