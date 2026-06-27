#!/usr/bin/env python3
"""
SPIKE: Extract per-rollout trajectory + object-pose data from OpenVLA on LIBERO.

Path taken: OpenVLA + LIBERO directly (NOT vla-evaluation-harness).
The harness only records reward/done/success per step — no EE trajectory or object
poses without forking LIBEROBenchmark. The OpenVLA eval loop already has obs access.

Setup (fresh CUDA env, Python 3.10 recommended):
  git clone https://github.com/openvla/openvla.git
  git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
  conda create -n openvla-spike python=3.10 -y && conda activate openvla-spike
  conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia -y
  pip install -e ./openvla
  pip install packaging ninja && pip install "flash-attn==2.5.5" --no-build-isolation
  pip install -e ./LIBERO
  pip install -r ./openvla/experiments/robot/libero/libero_requirements.txt

Run:
  python spike_run_libero.py [--num_episodes 20] [--load_in_4bit]

Outputs:
  spike_rollouts.csv
  spike_trajectories/seed_<N>.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent
OPENVLA_ROOT = ROOT / "openvla"
LIBERO_ROOT = ROOT / "LIBERO"

# OpenVLA imports expect CWD on openvla root and that path on sys.path.
sys.path.insert(0, str(LIBERO_ROOT))
sys.path.insert(0, str(OPENVLA_ROOT))
os.chdir(OPENVLA_ROOT)

from libero.libero import benchmark  # noqa: E402
from libero.libero import get_libero_path  # noqa: E402
from libero.libero.envs import OffScreenRenderEnv  # noqa: E402

from experiments.robot.libero.libero_utils import (  # noqa: E402
    get_libero_dummy_action,
    get_libero_image,
    quat2axisangle,
    resize_image,
)
from experiments.robot.openvla_utils import get_processor  # noqa: E402
from experiments.robot.robot_utils import (  # noqa: E402
    get_action,
    get_image_resize_size,
    get_model,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)


@dataclass
class SpikeConfig:
    task_suite_name: str = "libero_spatial"
    task_id: int = 0
    num_episodes: int = 20
    num_steps_wait: int = 10
    model_seed: int = 7
    env_seed: int = 0
    pretrained_checkpoint: str = "openvla/openvla-7b-finetuned-libero-spatial"
    center_crop: bool = True
    load_in_4bit: bool = False
    load_in_8bit: bool = False
    output_csv: str = str(ROOT / "spike_rollouts.csv")
    trajectory_dir: str = str(ROOT / "spike_trajectories")


MAX_STEPS_BY_SUITE = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}


def get_libero_env_with_seed(task, env_seed: int, resolution: int = 256):
    """Same as libero_utils.get_libero_env but exposes env_seed."""
    task_description = task.language
    task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(env_seed)
    return env, task_description


def get_pick_target_name(env) -> str:
    """First obj_of_interest entry is the manipulated object in LIBERO BDDL tasks."""
    return env.obj_of_interest[0]


def read_object_pos(obs: dict, obj_name: str) -> np.ndarray:
    key = f"{obj_name}_pos"
    if key not in obs:
        available = [k for k in obs if k.endswith("_pos") and not k.startswith("robot")]
        raise KeyError(f"{key} not in obs. Available object keys: {available}")
    return np.asarray(obs[key], dtype=np.float64)


def ee_pos_from_obs(obs: dict) -> list[float]:
    return np.asarray(obs["robot0_eef_pos"], dtype=np.float64).tolist()


def run_episode(cfg_obj, model, processor, env, task_description, initial_state, resize_size, max_steps):
    env.reset()
    obs = env.set_init_state(initial_state)

    target_name = get_pick_target_name(env)
    ee_trajectory: list[list[float]] = []
    init_pos: np.ndarray | None = None
    t = 0
    success = False

    while t < max_steps + cfg_obj.num_steps_wait:
        if t < cfg_obj.num_steps_wait:
            obs, _, done, _ = env.step(get_libero_dummy_action("openvla"))
            t += 1
            continue

        if init_pos is None:
            init_pos = read_object_pos(obs, target_name)

        # Record EE after settle, before policy step.
        ee_trajectory.append(ee_pos_from_obs(obs))

        img = get_libero_image(obs, resize_size)
        observation = {
            "full_image": img,
            "state": np.concatenate(
                (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
            ),
        }
        action = get_action(cfg_obj, model, observation, task_description, processor=processor)
        action = normalize_gripper_action(action, binarize=True)
        action = invert_gripper_action(action)

        obs, _, done, _ = env.step(action.tolist())
        if done:
            success = True
            ee_trajectory.append(ee_pos_from_obs(obs))
            t += 1
            break
        t += 1

    assert init_pos is not None
    num_policy_steps = max(0, t - cfg_obj.num_steps_wait)
    return {
        "target_object_name": target_name,
        "target_object_init_x": float(init_pos[0]),
        "target_object_init_y": float(init_pos[1]),
        "target_object_init_z": float(init_pos[2]),
        "ee_trajectory": ee_trajectory,
        "success": success,
        "num_steps": num_policy_steps,
    }


def write_trajectory_file(traj_dir: Path, episode_idx: int, ee_trajectory: list[list[float]]) -> str:
    traj_dir.mkdir(parents=True, exist_ok=True)
    path = traj_dir / f"seed_{episode_idx}.json"
    with open(path, "w") as f:
        json.dump(ee_trajectory, f)
    return str(path)


def summarize_position_variation(rows: list[dict]) -> dict[str, Any]:
    xs = [r["target_object_init_x"] for r in rows]
    ys = [r["target_object_init_y"] for r in rows]
    zs = [r["target_object_init_z"] for r in rows]
    def span(vals):
        return float(max(vals) - min(vals)) * 100.0  # meters -> cm
    return {
        "x_range_cm": span(xs),
        "y_range_cm": span(ys),
        "z_range_cm": span(zs),
        "x_std_cm": float(np.std(xs)) * 100.0,
        "y_std_cm": float(np.std(ys)) * 100.0,
        "z_std_cm": float(np.std(zs)) * 100.0,
    }


def parse_args() -> SpikeConfig:
    p = argparse.ArgumentParser(description="OpenVLA/LIBERO trajectory extraction spike")
    p.add_argument("--task_suite_name", default="libero_spatial")
    p.add_argument("--task_id", type=int, default=0)
    p.add_argument("--num_episodes", type=int, default=20)
    p.add_argument("--model_seed", type=int, default=7)
    p.add_argument("--env_seed", type=int, default=0)
    p.add_argument("--pretrained_checkpoint", default="openvla/openvla-7b-finetuned-libero-spatial")
    p.add_argument("--load_in_4bit", action="store_true")
    p.add_argument("--load_in_8bit", action="store_true")
    p.add_argument("--output_csv", default=str(ROOT / "spike_rollouts.csv"))
    p.add_argument("--trajectory_dir", default=str(ROOT / "spike_trajectories"))
    args = p.parse_args()
    return SpikeConfig(
        task_suite_name=args.task_suite_name,
        task_id=args.task_id,
        num_episodes=args.num_episodes,
        model_seed=args.model_seed,
        env_seed=args.env_seed,
        pretrained_checkpoint=args.pretrained_checkpoint,
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit,
        output_csv=args.output_csv,
        trajectory_dir=args.trajectory_dir,
    )


def main():
    cfg = parse_args()

    if not OPENVLA_ROOT.is_dir():
        raise SystemExit(f"OpenVLA repo not found at {OPENVLA_ROOT}")
    if not LIBERO_ROOT.is_dir():
        raise SystemExit(f"LIBERO repo not found at {LIBERO_ROOT}")

    import torch

    device = torch.device("cuda:0") if torch.cuda.is_available() else None
    if device is None and torch.backends.mps.is_available():
        device = torch.device("mps")
    if device is None:
        raise SystemExit(
            "CUDA GPU required for OpenVLA-7B inference. No CUDA or MPS device found. "
            "Run on a machine with an NVIDIA GPU and PyTorch CUDA installed."
        )

    # Patch OpenVLA utils for non-CUDA devices (dev fallback; CUDA recommended for speed).
    import experiments.robot.openvla_utils as openvla_utils
    import experiments.robot.robot_utils as robot_utils

    openvla_utils.DEVICE = device
    robot_utils.DEVICE = device

    _orig_get_vla = openvla_utils.get_vla

    def _get_vla_patched(cfg):
        import json
        import os
        import time
        from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor
        from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
        from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
        from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor

        print("[*] Instantiating Pretrained VLA model")
        attn_impl = "flash_attention_2" if device.type == "cuda" else "sdpa"
        dtype = torch.bfloat16 if device.type == "cuda" else torch.float16
        print(f"[*] device={device}, attn={attn_impl}, dtype={dtype}")

        AutoConfig.register("openvla", OpenVLAConfig)
        AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
        AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
        AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

        vla = AutoModelForVision2Seq.from_pretrained(
            cfg.pretrained_checkpoint,
            attn_implementation=attn_impl,
            torch_dtype=dtype,
            load_in_8bit=cfg.load_in_8bit,
            load_in_4bit=cfg.load_in_4bit,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        if not cfg.load_in_8bit and not cfg.load_in_4bit:
            vla = vla.to(device)
        dataset_statistics_path = os.path.join(cfg.pretrained_checkpoint, "dataset_statistics.json")
        if not os.path.isfile(dataset_statistics_path):
            try:
                from huggingface_hub import hf_hub_download

                dataset_statistics_path = hf_hub_download(
                    cfg.pretrained_checkpoint, "dataset_statistics.json"
                )
            except Exception:
                dataset_statistics_path = None
        if dataset_statistics_path and os.path.isfile(dataset_statistics_path):
            with open(dataset_statistics_path, "r") as f:
                vla.norm_stats = json.load(f)
        return vla

    if device.type != "cuda":
        openvla_utils.get_vla = _get_vla_patched
        robot_utils.get_vla = _get_vla_patched

        _infer_dtype = torch.float16
        def _get_vla_action_mps(vla, processor, base_vla_name, obs, task_label, unnorm_key, center_crop=False):
            import tensorflow as tf
            from PIL import Image

            image = Image.fromarray(obs["full_image"]).convert("RGB")
            if center_crop:
                batch_size = 1
                crop_scale = 0.9
                image_tf = tf.convert_to_tensor(np.array(image))
                orig_dtype = image_tf.dtype
                image_tf = tf.image.convert_image_dtype(image_tf, tf.float32)
                image_tf = openvla_utils.crop_and_resize(image_tf, crop_scale, batch_size)
                image_tf = tf.clip_by_value(image_tf, 0, 1)
                image_tf = tf.image.convert_image_dtype(image_tf, orig_dtype, saturate=True)
                image = Image.fromarray(image_tf.numpy()).convert("RGB")

            if "openvla-v01" in base_vla_name:
                prompt = (
                    f"{openvla_utils.OPENVLA_V01_SYSTEM_PROMPT} USER: What action should the robot take to "
                    f"{task_label.lower()}? ASSISTANT:"
                )
            else:
                prompt = f"In: What action should the robot take to {task_label.lower()}?\nOut:"

            inputs = processor(prompt, image).to(device, dtype=_infer_dtype)
            action = vla.predict_action(**inputs, unnorm_key=unnorm_key, do_sample=False)
            return action

        openvla_utils.get_vla_action = _get_vla_action_mps
        robot_utils.get_vla_action = _get_vla_action_mps

    set_seed_everywhere(cfg.model_seed)

    # Minimal GenerateConfig-compatible namespace for OpenVLA helpers.
    class ModelCfg:
        model_family = "openvla"
        pretrained_checkpoint = cfg.pretrained_checkpoint
        load_in_4bit = cfg.load_in_4bit
        load_in_8bit = cfg.load_in_8bit
        center_crop = cfg.center_crop
        task_suite_name = cfg.task_suite_name
        unnorm_key = cfg.task_suite_name
        num_steps_wait = cfg.num_steps_wait

    model_cfg = ModelCfg()
    print(f"[*] Loading OpenVLA checkpoint: {cfg.pretrained_checkpoint}")
    model = get_model(model_cfg)
    processor = get_processor(model_cfg)
    resize_size = get_image_resize_size(model_cfg)

    task_suite = benchmark.get_benchmark_dict()[cfg.task_suite_name]()
    task = task_suite.get_task(cfg.task_id)
    initial_states = task_suite.get_task_init_states(cfg.task_id)
    max_steps = MAX_STEPS_BY_SUITE[cfg.task_suite_name]

    env, task_description = get_libero_env_with_seed(task, cfg.env_seed)
    print(f"[*] Task {cfg.task_id}: {task_description}")
    print(f"[*] Running {cfg.num_episodes} episodes (episode_idx 0..{cfg.num_episodes - 1})")

    rows: list[dict] = []
    traj_dir = Path(cfg.trajectory_dir)

    for episode_idx in range(cfg.num_episodes):
        if episode_idx >= len(initial_states):
            print(f"[!] Only {len(initial_states)} init states available; stopping at episode {episode_idx}")
            break

        result = run_episode(
            model_cfg,
            model,
            processor,
            env,
            task_description,
            initial_states[episode_idx],
            resize_size,
            max_steps,
        )
        traj_path = write_trajectory_file(traj_dir, episode_idx, result["ee_trajectory"])
        row = {
            "seed": episode_idx,
            "task_id": cfg.task_id,
            "task_description": task_description,
            "target_object_name": result["target_object_name"],
            "target_object_init_x": result["target_object_init_x"],
            "target_object_init_y": result["target_object_init_y"],
            "target_object_init_z": result["target_object_init_z"],
            "ee_trajectory_path": traj_path,
            "final_ee_x": result["ee_trajectory"][-1][0] if result["ee_trajectory"] else math.nan,
            "final_ee_y": result["ee_trajectory"][-1][1] if result["ee_trajectory"] else math.nan,
            "final_ee_z": result["ee_trajectory"][-1][2] if result["ee_trajectory"] else math.nan,
            "success": result["success"],
            "num_steps": result["num_steps"],
        }
        rows.append(row)
        status = "SUCCESS" if result["success"] else "FAIL"
        print(
            f"  seed={episode_idx:2d}  {status}  steps={result['num_steps']:3d}  "
            f"target=({result['target_object_init_x']:.3f}, {result['target_object_init_y']:.3f}, "
            f"{result['target_object_init_z']:.3f})  traj_len={len(result['ee_trajectory'])}"
        )

    fieldnames = [
        "seed",
        "task_id",
        "task_description",
        "target_object_name",
        "target_object_init_x",
        "target_object_init_y",
        "target_object_init_z",
        "ee_trajectory_path",
        "final_ee_x",
        "final_ee_y",
        "final_ee_z",
        "success",
        "num_steps",
    ]
    with open(cfg.output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    variation = summarize_position_variation(rows)
    print(f"\n[*] Wrote {len(rows)} rows to {cfg.output_csv}")
    print(
        f"[*] Target object init position variation (cm): "
        f"x={variation['x_range_cm']:.1f}, y={variation['y_range_cm']:.1f}, z={variation['z_range_cm']:.1f}"
    )

    findings_path = ROOT / "SPIKE_FINDINGS.md"
    write_findings(findings_path, cfg, task_description, rows, variation)
    print(f"[*] Wrote findings to {findings_path}")


def write_findings(path: Path, cfg: SpikeConfig, task_description: str, rows: list[dict], variation: dict, blocker: str | None = None):
    n_success = sum(1 for r in rows if r["success"])
    blocker_section = blocker or "None encountered during this run."
    content = f"""# SPIKE Findings: OpenVLA + LIBERO Trajectory Extraction

## Path taken

**OpenVLA + LIBERO directly** (not allenai/vla-evaluation-harness).

Reason: The harness decouples model inference from sim via WebSocket and only records
`reward`, `done`, and `success` per step. Per-step EE trajectories and object poses are
available in the sim observation dict but are not exported. Extending the harness would
require subclassing `LIBEROBenchmark` plus Docker + model server — more plumbing than
instrumenting the existing OpenVLA eval loop, which already reads `robot0_eef_pos` each step.

## Setup

- Model: `{cfg.pretrained_checkpoint}` (OpenVLA-7B, LIBERO-Spatial finetune)
- Suite: `{cfg.task_suite_name}`, task_id={cfg.task_id}
- Task: {task_description}
- Episodes: {len(rows)} (episode_idx 0..{len(rows) - 1} selects LIBERO init states)
- Model seed: {cfg.model_seed}, env seed: {cfg.env_seed}

## Results summary

- Rollouts completed: {len(rows)}
- Successes: {n_success}/{len(rows)} ({100.0 * n_success / max(len(rows), 1):.1f}%)
- Output CSV: `{cfg.output_csv}`
- Trajectory files: `{cfg.trajectory_dir}/seed_*.json`

---

## 1. TRAJECTORY ACCESS

**Yes.**

Per-step end-effector position is in the LIBERO/robosuite observation dict every step:
`obs["robot0_eef_pos"]` → `(x, y, z)` world frame. The spike script appends this after
the settle period and before/after each policy step.

How: read from `obs` inside the rollout loop (same keys OpenVLA already uses for proprio).
Full paths stored as JSON lists in `{cfg.trajectory_dir}/`; CSV column `ee_trajectory_path`
points to each file. Also logged: `final_ee_x/y/z`, `num_steps`.

---

## 2. OBJECT POSE ACCESS

**Yes.**

After `env.set_init_state(...)` and the {cfg.num_steps_wait} settle steps, object positions
are in obs as `{{object_name}}_pos` (from robosuite object observables, enabled by default
via `use_object_obs=True`). Target object name comes from `env.obj_of_interest[0]` (the
manipulated object in LIBERO BDDL).

This run used target object: `{rows[0]["target_object_name"] if rows else "N/A"}`.

---

## 3. SEED VARIATION

**Yes** — varying `episode_idx` (selects different entries in `task_suite.get_task_init_states`)
changes initial object layout.

Rough position ranges across {len(rows)} episodes (target object, post-settle):

| Axis | Range (cm) | Std (cm) |
|------|------------|----------|
| x | {variation["x_range_cm"]:.2f} | {variation["x_std_cm"]:.2f} |
| y | {variation["y_range_cm"]:.2f} | {variation["y_std_cm"]:.2f} |
| z | {variation["z_range_cm"]:.2f} | {variation["z_std_cm"]:.2f} |

Note: LIBERO eval varies **episode_idx** (init state index), not `env.seed`. The OpenVLA
reference hardcodes `env.seed(0)`. `model_seed` only affects policy stochasticity
(OpenVLA uses greedy decoding here).

---

## Blockers

{blocker_section}
"""
    path.write_text(content)


if __name__ == "__main__":
    main()
