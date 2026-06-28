#!/usr/bin/env python3
"""Collect PatchMask behavioral scores (proxy from position sensitivity until live probe)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from patchmask_behavioral import build_local_proxy_scores

ROOT = Path(__file__).resolve().parent


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", type=int, default=0)
    args = parser.parse_args()
    pos_path = ROOT / ("action_sensitivity_task0.json" if args.task_id == 0 else f"action_sensitivity_task{args.task_id}.json")
    conditions = build_local_proxy_scores(pos_path)
    out = {
        "task_id": args.task_id,
        "suite": "libero_spatial",
        "n_conditions": len(conditions),
        "conditions": conditions,
    }
    path = ROOT / "patchmask_behavioral_scores.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"Wrote {path} ({len(conditions)} conditions)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
