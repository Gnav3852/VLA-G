#!/usr/bin/env python3
"""
Controlled-perturbation Pi0.5/LIBERO rollout sweep on Modal.

Run one de-risking rollout:
  modal run modal_pi05_sweep.py --test

Run the tight-grid Phase 2 sweep (5x5 x 3 repeats = 75 rollouts):
  modal run modal_pi05_sweep.py

Task 2 replication:
  modal run modal_pi05_sweep.py --task-id 2 --test
  modal run modal_pi05_sweep.py --task-id 2

Outputs written locally (task_id selects paths via libero_task_config.py):
  pi05_rollouts.csv / pi05_task2_rollouts.csv + trajectories/
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import modal

APP_NAME = "vla-pi05-perturbation-sweep"
OPENPI_REV = "981483dca0fd9acba698fea00aa6e52d56a66c58"
CONFIG_NAME = "pi05_libero"
CHECKPOINT_GCS = f"gs://openpi-assets/checkpoints/{CONFIG_NAME}"
CHECKPOINT_PATH = f"/weights/{CONFIG_NAME}"
GPU_TYPE = "A100"

LOCAL_ROOT = Path(__file__).resolve().parent

TASK_SUITE = "libero_spatial"
TARGET_OBJECT = "akita_black_bowl_1"
MODEL_SEED = 7
ENV_SEED = 0
NUM_STEPS_WAIT = 10
MAX_STEPS = 220
CHUNK_SIZE = 10
IMAGE_SIZE = 224
GRID_HALF_WIDTH = 0.07
GRID_SIZE = 5
N_REPEATS = 3
POSITION_TOLERANCE_M = 0.015

app = modal.App(APP_NAME)
weights_volume = modal.Volume.from_name("model-weights-pi05", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
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
            "PYTHONPATH": "/opt/openpi/src:/opt/LIBERO",
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
            "XLA_PYTHON_CLIENT_ALLOCATOR": "platform",
        }
    )
    .pip_install("packaging", "wheel", "setuptools", "ninja", "hatchling")
    .pip_install(
        "jax[cuda12]==0.5.3",
        "flax==0.10.2",
        "orbax-checkpoint==0.11.13",
        "chex",
        "pytest",
        "einops>=0.8.0",
        "equinox>=0.11.8",
        "augmax>=0.3.4",
        "dm-tree>=0.1.8",
        "flatbuffers>=24.3.25",
        "fsspec[gcs]>=2024.6.0",
        "gcsfs>=2024.6.0",
        "imageio>=2.36.1",
        "jaxtyping==0.2.36",
        "ml_collections==1.0.0",
        "numpy>=1.22.4,<2.0.0",
        "numpydantic>=1.6.6",
        "opencv-python>=4.10.0.84",
        "pillow>=11.0.0",
        "sentencepiece>=0.2.0",
        "torch==2.7.1",
        "transformers==4.53.2",
        "typing-extensions>=4.12.2",
        "tyro>=0.9.5",
        "wandb>=0.19.1",
        "filelock>=3.16.1",
        "beartype==0.19.0",
        "treescope>=0.1.7",
        "rich>=14.0.0",
        "polars>=1.30.0",
        "tqdm-loggable>=0.2",
        "ml-dtypes==0.4.1",
        "tensorstore==0.1.74",
    )
    .pip_install(
        "PyOpenGL",
        "bddl",
        "cloudpickle",
        "easydict",
        "gym",
        "imageio[ffmpeg]",
        "matplotlib",
        "mujoco==2.3.7",
        "robosuite==1.4.1",
        "pyyaml",
        "hydra-core",
        "future",
        "scipy",
    )
    .run_commands(
        "git clone https://github.com/Physical-Intelligence/openpi.git /opt/openpi",
        f"cd /opt/openpi && git checkout {OPENPI_REV}",
        "pip install -e /opt/openpi/packages/openpi-client",
        "pip install git+https://github.com/huggingface/lerobot@0cf864870cf29f4738d3ade893e6fd13fbd7cdb5",
        "pip install -e /opt/openpi --no-deps",
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


@app.function(image=image, volumes={"/weights": weights_volume}, gpu=GPU_TYPE, timeout=3600)
def prep_weights() -> str:
    from pathlib import Path

    import fsspec

    dest = Path(CHECKPOINT_PATH)
    if dest.exists() and any(dest.iterdir()):
        print(f"Checkpoint already present at {CHECKPOINT_PATH}")
        weights_volume.commit()
        return CHECKPOINT_PATH

    print(f"Downloading {CHECKPOINT_GCS} -> {CHECKPOINT_PATH}")
    dest.mkdir(parents=True, exist_ok=True)
    fs = fsspec.filesystem("gs", token="anon")
    fs.get(CHECKPOINT_GCS.rstrip("/") + "/", str(dest), recursive=True)
    weights_volume.commit()
    return CHECKPOINT_PATH


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


@app.cls(
    image=image,
    volumes={"/weights": weights_volume},
    gpu=GPU_TYPE,
    timeout=1200,
    retries=0,
    max_containers=4,
    scaledown_window=300,
)
class Pi05RolloutWorker:
    """Load Pi0.5 once per container, then run LIBERO rollouts."""

    @modal.enter()
    def load_policy(self) -> None:
        import os
        import sys

        os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
        os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
        sys.path.insert(0, "/opt/openpi/src")

        import jax

        jax.config.update("jax_platform_name", "gpu")

        from openpi.policies import policy_config
        from openpi.training import config as openpi_config

        cfg = openpi_config.get_config(CONFIG_NAME)
        self.policy = policy_config.create_trained_policy(cfg, CHECKPOINT_PATH)
        print(f"Pi0.5 policy loaded from {CHECKPOINT_PATH}")

    @modal.method()
    def run_rollout(self, rollout_id: int, grid_x: float, grid_y: float, base_idx: int, task_id: int) -> dict[str, Any]:
        import math
        import os
        import sys
        from collections import deque

        import numpy as np
        import torch
        from openpi_client import image_tools

        _orig_torch_load = torch.load

        def _torch_load_compat(*args, **kwargs):
            kwargs.setdefault("weights_only", False)
            return _orig_torch_load(*args, **kwargs)

        torch.load = _torch_load_compat

        os.environ.setdefault("MUJOCO_GL", "egl")
        os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
        os.environ.setdefault("NVIDIA_DRIVER_CAPABILITIES", "all")
        sys.path.insert(0, "/opt/LIBERO")

        from libero.libero import benchmark, get_libero_path
        from libero.libero.envs import OffScreenRenderEnv

        LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]

        def quat2axisangle(quat: np.ndarray) -> np.ndarray:
            q = np.asarray(quat, dtype=np.float64)
            if q[3] > 1.0:
                q[3] = 1.0
            elif q[3] < -1.0:
                q[3] = -1.0
            den = np.sqrt(1.0 - q[3] * q[3])
            if math.isclose(float(den), 0.0):
                return np.zeros(3, dtype=np.float64)
            return (q[:3] * 2.0 * math.acos(q[3])) / den

        def build_openpi_obs(obs, task_description: str) -> dict[str, Any]:
            img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
            wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
            img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, IMAGE_SIZE, IMAGE_SIZE))
            wrist = image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist, IMAGE_SIZE, IMAGE_SIZE))
            state = np.concatenate(
                [
                    obs["robot0_eef_pos"],
                    quat2axisangle(obs["robot0_eef_quat"]),
                    obs["robot0_gripper_qpos"],
                ],
                dtype=np.float64,
            )
            return {
                "observation/image": img,
                "observation/wrist_image": wrist,
                "observation/state": state,
                "prompt": str(task_description),
            }

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
            qpos_start = 1 + start
            patched_state[qpos_start] = x
            patched_state[qpos_start + 1] = y
            return patched_state, float(patched_state[qpos_start + 2])

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

        env = None
        try:
            print(f"[rollout {rollout_id}] starting grid=({grid_x:.3f}, {grid_y:.3f}) base_idx={base_idx}")
            np.random.seed(MODEL_SEED)

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
            action_plan: deque[np.ndarray] = deque()

            while t < MAX_STEPS + NUM_STEPS_WAIT:
                if t < NUM_STEPS_WAIT:
                    obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)
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

                if not action_plan:
                    openpi_obs = build_openpi_obs(obs, task_description)
                    chunk = np.asarray(self.policy.infer(openpi_obs)["actions"], dtype=np.float32)
                    if chunk.ndim == 1:
                        chunk = chunk.reshape(1, -1)
                    if chunk.shape[0] < CHUNK_SIZE:
                        raise RuntimeError(f"Policy returned {chunk.shape[0]} actions, need {CHUNK_SIZE}")
                    action_plan.extend(chunk[:CHUNK_SIZE])

                action = np.asarray(action_plan.popleft(), dtype=np.float64)
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
            out = dict(base_result)
            out.update(
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
                f"[rollout {rollout_id}] done success={success} steps={out['num_steps']} "
                f"traj_len={len(trajectory)}"
            )
            env.close()
            return out
        except Exception as exc:
            base_result["error"] = repr(exc)
            print(f"[rollout {rollout_id}] ERROR: {exc!r}")
            if env is not None:
                try:
                    env.close()
                except Exception:
                    pass
            return base_result


@app.local_entrypoint()
def main(test: bool = False, skip_prep: bool = False, task_id: int = 0) -> None:
    from libero_task_config import get_task_preset

    preset = get_task_preset(task_id)
    print(f"Task {task_id}: {preset.instruction}")
    print(f"Grid center=({preset.center_x:.3f}, {preset.center_y:.3f}), target={preset.target_object}")

    if not skip_prep:
        print("Ensuring Pi0.5 checkpoint is cached on Modal Volume...")
        prep_weights.remote()

    configs = (
        [(0, preset.center_x, preset.center_y, 0, task_id)]
        if test
        else [(rid, gx, gy, bi, task_id) for rid, gx, gy, bi in build_grid(preset.center_x, preset.center_y)]
    )
    est_rollout_seconds = 120
    est_a100_dollars_per_hour = 2.10
    est_cost = len(configs) * est_rollout_seconds / 3600.0 * est_a100_dollars_per_hour
    print(f"Running {len(configs)} rollout(s) on {GPU_TYPE}; rough compute estimate: ${est_cost:.2f}")

    results = list(Pi05RolloutWorker().run_rollout.starmap(configs, order_outputs=False))
    errors = [r for r in results if r.get("status") != "ok"]
    if errors:
        print("Some rollouts failed; keeping rows with error details:")
        for row in sorted(errors, key=lambda r: r["rollout_id"]):
            print(f"  rollout {row['rollout_id']}: {row['error']}")

    _write_local_outputs(results, preset.pi05_csv, preset.pi05_traj_dir)
