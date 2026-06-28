#!/usr/bin/env python3
"""
Orchestrate wide-range position sweeps (FINAL RUN).

Runs OpenVLA + Pi0.5 at ±10/15/20 cm on task 0 via Modal.

  ./.venv/bin/python run_wide_position_sweep.py --test
  ./.venv/bin/python run_wide_position_sweep.py
  ./.venv/bin/python run_wide_position_sweep.py --half-widths 0.15 0.20 --models openvla
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from libero_task_config import WIDE_HALF_WIDTHS_M, get_task_preset, wide_sweep_paths

ROOT = Path(__file__).resolve().parent
MODAL = ROOT / ".venv" / "bin" / "modal"
WIDE_N_REPEATS = 5


def run_cmd(cmd: list[str]) -> int:
    print("\n>>> " + " ".join(cmd), flush=True)
    return subprocess.call(cmd, cwd=ROOT)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run wide-range position sweeps on Modal")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--models", nargs="+", default=["openvla", "pi05"], choices=["openvla", "pi05"])
    parser.add_argument("--half-widths", type=float, nargs="+", default=list(WIDE_HALF_WIDTHS_M))
    parser.add_argument("--n-repeats", type=int, default=WIDE_N_REPEATS)
    parser.add_argument("--test", action="store_true", help="Single center rollout per combo")
    parser.add_argument("--skip-prep", action="store_true")
    args = parser.parse_args()

    if not MODAL.is_file():
        print(f"Missing Modal CLI: {MODAL}", file=sys.stderr)
        return 1

    preset = get_task_preset(args.task_id)
    print(f"Wide sweep task {args.task_id}: center=({preset.center_x}, {preset.center_y})")
    print(f"Half-widths: {[f'±{w*100:.0f}cm' for w in args.half_widths]}, repeats={args.n_repeats}")

    rc = 0
    skip_prep = args.skip_prep
    for half in args.half_widths:
        for model in args.models:
            csv_path, traj_dir = wide_sweep_paths(args.task_id, model, half)
            script = "modal_perturbation_sweep.py" if model == "openvla" else "modal_pi05_sweep.py"
            cmd = [
                str(MODAL),
                "run",
                script,
                f"--task-id={args.task_id}",
                f"--grid-half-width={half}",
                f"--n-repeats={args.n_repeats}",
                f"--output-csv={csv_path}",
                f"--output-traj-dir={traj_dir}",
            ]
            if args.test:
                cmd.append("--test")
            if skip_prep:
                cmd.append("--skip-prep")
            skip_prep = True  # weights cached after first run
            if run_cmd(cmd) != 0:
                rc = 1
                print(f"FAILED: {model} ±{half*100:.0f}cm", file=sys.stderr)
    return rc


if __name__ == "__main__":
    sys.exit(main())
