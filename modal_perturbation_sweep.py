#!/usr/bin/env python3
"""
Controlled-perturbation OpenVLA/LIBERO rollout sweep on Modal.

Run one de-risking rollout:
  modal run modal_perturbation_sweep.py --test

Run the tight-grid Phase 1 sweep (5x5 x 3 repeats = 75 rollouts):
  modal run modal_perturbation_sweep.py

Task 2 replication:
  modal run modal_perturbation_sweep.py --task-id 2 --test
  modal run modal_perturbation_sweep.py --task-id 2

Outputs written locally (task_id selects paths via libero_task_config.py):
  spike3_rollouts.csv / openvla_task2_rollouts.csv + trajectories/
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import modal

APP_NAME = "vla-shortcut-perturbation-sweep"
MODEL_ID = "openvla/openvla-7b-finetuned-libero-spatial"
WEIGHTS_PATH = "/weights/openvla-7b-finetuned-libero-spatial"
GPU_TYPE = "A10"  # Modal's current slug for A10G.

LOCAL_ROOT = Path(__file__).resolve().parent

TASK_SUITE = "libero_spatial"
TARGET_OBJECT = "akita_black_bowl_1"
MODEL_SEED = 7
ENV_SEED = 0
NUM_STEPS_WAIT = 10
MAX_STEPS = 220
GRID_HALF_WIDTH = 0.07
GRID_SIZE = 5
N_REPEATS = 3
POSITION_TOLERANCE_M = 0.015

app = modal.App(APP_NAME)
weights_volume = modal.Volume.from_name("model-weights-openvla", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install(
        "build-essential",
        "git",
        "libglib2.0-0",
        "libegl1",
        "libegl1-mesa",
        "libgl1",
        "libgl1-mesa-glx",
        "libgles2",
        "libglvnd-dev",
        "libglvnd0",
        "libosmesa6-dev",
        "mesa-utils",
        "ninja-build",
        "xvfb",
    )
    .env(
        {
            "MUJOCO_GL": "egl",
            "PYOPENGL_PLATFORM": "egl",
            "NVIDIA_DRIVER_CAPABILITIES": "all",
            "PYTHONPATH": "/opt/openvla:/opt/LIBERO",
            "HF_HOME": "/weights/hf-cache",
            "TRANSFORMERS_CACHE": "/weights/hf-cache",
            "TORCH_HOME": "/weights/torch-cache",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        }
    )
    .pip_install("packaging", "wheel", "setuptools", "ninja")
    .pip_install(
        "torch==2.2.0",
        "torchvision==0.17.0",
        "torchaudio==2.2.0",
        index_url="https://download.pytorch.org/whl/cu121",
    )
    .pip_install(
        "accelerate>=0.25.0",
        "draccus==0.8.0",
        "einops",
        "huggingface_hub<1.0",
        "json-numpy",
        "jsonlines",
        "matplotlib",
        "numpy<2",
        "peft==0.11.1",
        "protobuf==3.20.3",
        "rich",
        "safetensors",
        "sentencepiece==0.1.99",
        "tensorflow==2.15.0",
        "tensorflow_datasets==4.9.3",
        "tensorflow_graphics==2021.12.3",
        "tensorflow_metadata==1.14.0",
        "timm==0.9.10",
        "tokenizers==0.19.1",
        "transformers==4.40.1",
        "wandb",
    )
    .pip_install(
        "PyOpenGL",
        "bddl",
        "cloudpickle",
        "easydict",
        "gym",
        "imageio[ffmpeg]",
        "mujoco==2.3.7",
        "robosuite==1.4.1",
    )
    .pip_install("numpy<2", "opencv-python==4.9.0.80")
    .run_commands(
        "git clone --depth 1 https://github.com/openvla/openvla.git /opt/openvla",
        "pip install -e /opt/openvla --no-deps",
        "pip install git+https://github.com/moojink/dlimp_openvla",
        "git clone --depth 1 https://github.com/Lifelong-Robot-Learning/LIBERO.git /opt/LIBERO",
        "pip install -e /opt/LIBERO --no-deps",
    )
    .run_commands(
        "mkdir -p /root/.libero",
        "python - <<'PY'\n"
        "from pathlib import Path\n"
        "import yaml\n"
        "Path('/root/.libero').mkdir(parents=True, exist_ok=True)\n"
        "cfg = {\n"
        "    'benchmark_root': '/opt/LIBERO/libero/libero',\n"
        "    'bddl_files': '/opt/LIBERO/libero/libero/bddl_files',\n"
        "    'init_states': '/opt/LIBERO/libero/libero/init_files',\n"
        "    'datasets': '/opt/LIBERO/libero/datasets',\n"
        "    'assets': '/opt/LIBERO/libero/libero/assets',\n"
        "}\n"
        "Path('/root/.libero/config.yaml').write_text(yaml.safe_dump(cfg))\n"
        "PY",
    )
)


def build_grid(center_x: float, center_y: float) -> list[tuple[int, float, float, int]]:
    import numpy as np

    xs = np.linspace(center_x - GRID_HALF_WIDTH, center_x + GRID_HALF_WIDTH, GRID_SIZE)
    ys = np.linspace(center_y - GRID_HALF_WIDTH, center_y + GRID_HALF_WIDTH, GRID_SIZE)
    configs: list[tuple[int, float, float, int]] = []
    rollout_id = 0
    for x in xs:
        for y in ys:
            for base_idx in range(N_REPEATS):
                configs.append((rollout_id, float(x), float(y), base_idx))
                rollout_id += 1
    return configs


@app.function(image=image, volumes={"/weights": weights_volume}, gpu=GPU_TYPE, timeout=1800)
def prep_weights() -> str:
    from pathlib import Path

    from huggingface_hub import snapshot_download

    local_dir = Path(WEIGHTS_PATH)
    local_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading/caching {MODEL_ID} into {WEIGHTS_PATH}")
    snapshot_download(MODEL_ID, local_dir=WEIGHTS_PATH, local_dir_use_symlinks=False)
    weights_volume.commit()
    return WEIGHTS_PATH


def _write_local_outputs(results: list[dict[str, Any]], output_csv: Path, trajectory_dir: Path) -> None:
    trajectory_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    for result in sorted(results, key=lambda r: r["rollout_id"]):
        trajectory = result.pop("ee_trajectory")
        traj_path = trajectory_dir / f"rollout_{result['rollout_id']:03d}.json"
        traj_path.write_text(json.dumps(trajectory))
        result["ee_trajectory_path"] = str(traj_path)
        rows.append(result)

    fieldnames = [
        "rollout_id",
        "grid_x",
        "grid_y",
        "base_idx",
        "actual_object_init_x",
        "actual_object_init_y",
        "actual_object_init_z",
        "override_error_m",
        "grasp_point_x",
        "grasp_point_y",
        "grasp_point_z",
        "success",
        "num_steps",
        "ee_trajectory_path",
        "status",
        "error",
    ]
    with output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {output_csv}")
    print(f"Wrote trajectories to {trajectory_dir}")


@app.function(
    image=image,
    volumes={"/weights": weights_volume},
    gpu=GPU_TYPE,
    timeout=900,
    retries=1,
    max_containers=6,
)
def run_rollout(rollout_id: int, grid_x: float, grid_y: float, base_idx: int, task_id: int) -> dict[str, Any]:
    import os
    import sys
    from dataclasses import dataclass
    from pathlib import Path
    import gc

    import numpy as np
    import torch

    # These must be set before importing robosuite / MuJoCo.
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.setdefault("NVIDIA_DRIVER_CAPABILITIES", "all")
    sys.path.insert(0, "/opt/LIBERO")
    sys.path.insert(0, "/opt/openvla")
    os.chdir("/opt/openvla")

    def cleanup_cuda():
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except RuntimeError:
                pass

    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    from experiments.robot.libero.libero_utils import (
        get_libero_dummy_action,
        get_libero_image,
        quat2axisangle,
    )
    import experiments.robot.openvla_utils as openvla_utils
    import experiments.robot.robot_utils as robot_utils
    from experiments.robot.openvla_utils import get_processor
    from experiments.robot.robot_utils import (
        get_action,
        get_image_resize_size,
        get_model,
        invert_gripper_action,
        normalize_gripper_action,
        set_seed_everywhere,
    )

    def patch_openvla_loader_for_sdpa():
        import json

        from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
        from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
        from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
        from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor

        def get_vla_sdpa(cfg):
            print("[*] Instantiating OpenVLA with SDPA attention")
            AutoConfig.register("openvla", OpenVLAConfig)
            AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
            AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
            AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

            vla = AutoModelForVision2Seq.from_pretrained(
                cfg.pretrained_checkpoint,
                attn_implementation="sdpa",
                torch_dtype=torch.bfloat16,
                load_in_8bit=cfg.load_in_8bit,
                load_in_4bit=cfg.load_in_4bit,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            )
            if not cfg.load_in_8bit and not cfg.load_in_4bit:
                vla = vla.to(torch.device("cuda:0"))

            stats_path = Path(cfg.pretrained_checkpoint) / "dataset_statistics.json"
            if stats_path.is_file():
                with stats_path.open() as f:
                    vla.norm_stats = json.load(f)
            return vla

        openvla_utils.get_vla = get_vla_sdpa
        robot_utils.get_vla = get_vla_sdpa

    @dataclass
    class ModelCfg:
        model_family: str = "openvla"
        pretrained_checkpoint: str = WEIGHTS_PATH
        load_in_8bit: bool = False
        load_in_4bit: bool = False
        center_crop: bool = True
        task_suite_name: str = TASK_SUITE
        unnorm_key: str = TASK_SUITE
        num_steps_wait: int = NUM_STEPS_WAIT

    def get_env(task):
        task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
        env_args = {
            "bddl_file_name": task_bddl_file,
            "camera_heights": 256,
            "camera_widths": 256,
        }
        env = OffScreenRenderEnv(**env_args)
        env.seed(ENV_SEED)
        return env, task.language

    def patch_target_object_xy(env, state, obj_name: str, x: float, y: float):
        patched_state = np.asarray(state, dtype=np.float64).copy()
        obj = env.env.get_object(obj_name)
        joint_name = obj.joints[-1]
        addr = env.sim.model.get_joint_qpos_addr(joint_name)
        if isinstance(addr, tuple):
            start, end = addr
        else:
            start, end = addr, addr + 1
        if end - start < 7:
            raise ValueError(f"{obj_name} joint {joint_name} is not a free joint; qpos addr={addr}")

        # Flat state layout is [time, qpos..., qvel...]. Keep z and quaternion from the base state.
        qpos_start = 1 + start
        patched_state[qpos_start] = x
        patched_state[qpos_start + 1] = y
        commanded_z = float(patched_state[qpos_start + 2])
        return patched_state, commanded_z

    def ee_pos(obs) -> list[float]:
        return np.asarray(obs["robot0_eef_pos"], dtype=np.float64).tolist()

    def read_object_pos(obs, obj_name: str) -> np.ndarray:
        key = f"{obj_name}_pos"
        if key not in obs:
            available = [k for k in obs if k.endswith("_pos") and not k.startswith("robot")]
            raise KeyError(f"{key} not in obs. Available object keys: {available}")
        return np.asarray(obs[key], dtype=np.float64)

    def grasp_point(trajectory: list[list[float]]) -> list[float]:
        arr = np.asarray(trajectory, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[0] == 0:
            return [float("nan"), float("nan"), float("nan")]
        idx = int(np.where(arr[:, 2] == arr[:, 2].min())[0][0])
        return arr[idx].tolist()

    base_result = {
        "rollout_id": rollout_id,
        "grid_x": grid_x,
        "grid_y": grid_y,
        "base_idx": base_idx,
        "actual_object_init_x": float("nan"),
        "actual_object_init_y": float("nan"),
        "actual_object_init_z": float("nan"),
        "override_error_m": float("nan"),
        "grasp_point_x": float("nan"),
        "grasp_point_y": float("nan"),
        "grasp_point_z": float("nan"),
        "success": False,
        "num_steps": 0,
        "ee_trajectory": [],
        "status": "error",
        "error": "",
    }

    try:
        print(
            f"[rollout {rollout_id}] starting grid=({grid_x:.3f}, {grid_y:.3f}) base_idx={base_idx}"
        )
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available inside Modal container")

        cleanup_cuda()
        patch_openvla_loader_for_sdpa()
        set_seed_everywhere(MODEL_SEED)
        model_cfg = ModelCfg()
        model = get_model(model_cfg)
        processor = get_processor(model_cfg)
        resize_size = get_image_resize_size(model_cfg)

        task_suite = benchmark.get_benchmark_dict()[TASK_SUITE]()
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        episode_idx = int(base_idx) % len(initial_states)
        env, task_description = get_env(task)

        env.reset()
        patched_state, _ = patch_target_object_xy(
            env=env,
            state=initial_states[episode_idx],
            obj_name=TARGET_OBJECT,
            x=grid_x,
            y=grid_y,
        )
        obs = env.set_init_state(patched_state)

        t = 0
        actual_init = None
        trajectory: list[list[float]] = []
        success = False

        while t < MAX_STEPS + NUM_STEPS_WAIT:
            if t < NUM_STEPS_WAIT:
                obs, _, _, _ = env.step(get_libero_dummy_action("openvla"))
                t += 1
                continue

            if actual_init is None:
                actual_init = read_object_pos(obs, TARGET_OBJECT)
                err = float(np.linalg.norm(actual_init[:2] - np.asarray([grid_x, grid_y], dtype=np.float64)))
                print(
                    f"[rollout {rollout_id}] actual=({actual_init[0]:.3f}, {actual_init[1]:.3f}, "
                    f"{actual_init[2]:.3f}) override_error={err:.4f}m"
                )
                if err > POSITION_TOLERANCE_M:
                    raise RuntimeError(
                        f"Object override did not take cleanly: commanded=({grid_x:.3f},{grid_y:.3f}), "
                        f"actual=({actual_init[0]:.3f},{actual_init[1]:.3f}), err={err:.4f}m"
                    )

            trajectory.append(ee_pos(obs))
            img = get_libero_image(obs, resize_size)
            observation = {
                "full_image": img,
                "state": np.concatenate(
                    (
                        obs["robot0_eef_pos"],
                        quat2axisangle(obs["robot0_eef_quat"]),
                        obs["robot0_gripper_qpos"],
                    )
                ),
            }
            action = get_action(model_cfg, model, observation, task_description, processor=processor)
            action = normalize_gripper_action(action, binarize=True)
            action = invert_gripper_action(action)
            obs, _, done, _ = env.step(action.tolist())

            if done:
                success = True
                trajectory.append(ee_pos(obs))
                t += 1
                break
            t += 1

        if actual_init is None:
            actual_init = read_object_pos(obs, TARGET_OBJECT)
        gp = grasp_point(trajectory)
        result = dict(base_result)
        result.update(
            {
                "actual_object_init_x": float(actual_init[0]),
                "actual_object_init_y": float(actual_init[1]),
                "actual_object_init_z": float(actual_init[2]),
                "override_error_m": float(np.linalg.norm(actual_init[:2] - np.asarray([grid_x, grid_y]))),
                "grasp_point_x": float(gp[0]),
                "grasp_point_y": float(gp[1]),
                "grasp_point_z": float(gp[2]),
                "success": bool(success),
                "num_steps": max(0, t - NUM_STEPS_WAIT),
                "ee_trajectory": trajectory,
                "status": "ok",
                "error": "",
            }
        )
        print(
            f"[rollout {rollout_id}] done success={success} steps={result['num_steps']} "
            f"traj_len={len(trajectory)}"
        )
        env.close()
        del model
        del processor
        del env
        cleanup_cuda()
        return result
    except Exception as exc:
        base_result["error"] = repr(exc)
        print(f"[rollout {rollout_id}] ERROR: {exc!r}")
        try:
            env.close()
        except Exception:
            pass
        try:
            del model
        except Exception:
            pass
        try:
            del processor
        except Exception:
            pass
        cleanup_cuda()
        return base_result


@app.local_entrypoint()
def main(test: bool = False, skip_prep: bool = False, task_id: int = 0) -> None:
    from libero_task_config import get_task_preset

    preset = get_task_preset(task_id)
    print(f"Task {task_id}: {preset.instruction}")
    print(f"Grid center=({preset.center_x:.3f}, {preset.center_y:.3f}), target={preset.target_object}")

    if not skip_prep:
        print("Ensuring OpenVLA weights are cached on Modal Volume...")
        prep_weights.remote()

    configs = (
        [(0, preset.center_x, preset.center_y, 0, task_id)]
        if test
        else [(rid, gx, gy, bi, task_id) for rid, gx, gy, bi in build_grid(preset.center_x, preset.center_y)]
    )
    est_rollout_seconds = 90
    est_a10_dollars_per_hour = 0.36
    est_cost = len(configs) * est_rollout_seconds / 3600.0 * est_a10_dollars_per_hour
    print(f"Running {len(configs)} rollout(s) on {GPU_TYPE}; rough compute estimate: ${est_cost:.2f}")

    results = list(run_rollout.starmap(configs, order_outputs=False))
    errors = [r for r in results if r.get("status") != "ok"]
    if errors:
        print("Some rollouts failed; keeping rows with error details:")
        for row in sorted(errors, key=lambda r: r["rollout_id"]):
            print(f"  rollout {row['rollout_id']}: {row['error']}")

    _write_local_outputs(results, preset.openvla_csv, preset.openvla_traj_dir)
