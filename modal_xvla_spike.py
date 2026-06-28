#!/usr/bin/env python3
"""
X-VLA runnability spike on Modal/LIBERO via LeRobot xvla-libero checkpoint.

  ./.venv/bin/modal run modal_xvla_spike.py

Writes xvla_spike_result.json locally.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import modal

APP_NAME = "vla-xvla-spike"
CHECKPOINT = "lerobot/xvla-libero"
GPU_TYPE = "A10"
LOCAL_ROOT = Path(__file__).resolve().parent

TASK_SUITE = "libero_spatial"
TARGET_OBJECT = "akita_black_bowl_1"
CENTER_X = -0.06
CENTER_Y = 0.20

app = modal.App(APP_NAME)
weights_volume = modal.Volume.from_name("model-weights-xvla", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(
        "build-essential", "git", "libglib2.0-0", "libegl1", "libgl1", "libgl1-mesa-glx",
        "libgles2", "libglvnd0", "libosmesa6-dev", "xvfb",
    )
    .env(
        {
            "MUJOCO_GL": "egl",
            "PYOPENGL_PLATFORM": "egl",
            "HF_HOME": "/weights/hf-cache",
            "PYTHONPATH": "/opt/LIBERO",
        }
    )
    .pip_install("packaging", "wheel", "setuptools", "torch", "torchvision", "transformers", "accelerate", "numpy", "pillow")
    .pip_install("lerobot")
    .run_commands(
        "git clone --depth 1 https://github.com/Lifelong-Robot-Learning/LIBERO.git /opt/LIBERO",
        "pip install -e /opt/LIBERO --no-deps",
        "pip install mujoco==2.3.7 robosuite==1.4.1 opencv-python==4.9.0.80",
        "mkdir -p /root/.libero",
        "python - <<'PY'\nfrom pathlib import Path\nimport yaml\ncfg = {\n    'benchmark_root': '/opt/LIBERO/libero/libero',\n    'bddl_files': '/opt/LIBERO/libero/libero/bddl_files',\n    'init_states': '/opt/LIBERO/libero/libero/init_files',\n    'datasets': '/opt/LIBERO/libero/datasets',\n    'assets': '/opt/LIBERO/libero/libero/assets',\n}\nPath('/root/.libero/config.yaml').write_text(yaml.safe_dump(cfg))\nPY",
    )
)


def patch_target_object_xy(env, state, obj_name: str, x: float, y: float):
    import numpy as np

    patched_state = np.asarray(state, dtype=np.float64).copy()
    obj = env.env.get_object(obj_name)
    joint_name = obj.joints[-1]
    addr = env.sim.model.get_joint_qpos_addr(joint_name)
    if isinstance(addr, tuple):
        start, _end = addr
    else:
        start = addr
    qpos_start = 1 + start
    patched_state[qpos_start] = x
    patched_state[qpos_start + 1] = y
    return patched_state, float(patched_state[qpos_start + 2])


@app.function(image=image, gpu=GPU_TYPE, timeout=60 * 45, volumes={"/weights": weights_volume})
def run_spike(task_id: int, grid_x: float, grid_y: float) -> dict[str, Any]:
    import torch
    from libero.libero import benchmark

    result: dict[str, Any] = {
        "model": "x_vla",
        "checkpoint": CHECKPOINT,
        "status": "error",
        "success": False,
        "num_steps": 0,
        "error": "",
        "note": "Spike loads policy + LIBERO env; full obs adapter may need tuning.",
    }
    try:
        from lerobot.common.policies.xvla.modeling_xvla import XVLAPolicy
        from experiments.robot.robot_utils import get_libero_env, get_libero_dummy_action

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable")

        device = torch.device("cuda")
        policy = XVLAPolicy.from_pretrained(CHECKPOINT).to(device).eval()
        task_suite = benchmark.get_benchmark_dict()[TASK_SUITE]()
        task = task_suite.get_task(task_id)
        init_states = task_suite.get_task_init_states(task_id)
        env, task_description = get_libero_env(task, "openvla")
        env.reset()
        patched, _ = patch_target_object_xy(env, init_states[0], TARGET_OBJECT, grid_x, grid_y)
        obs = env.set_init_state(patched)

        steps = 0
        success = False
        for t in range(230):
            if t < 10:
                obs, _, _, _ = env.step(get_libero_dummy_action("openvla"))
                continue
            batch = {
                "observation.images.image": torch.zeros(1, 3, 224, 224, device=device),
                "observation.state": torch.zeros(1, 8, device=device),
                "task": [task_description],
            }
            with torch.no_grad():
                action = policy.select_action(batch)
            act = action[0].detach().cpu().numpy().tolist()
            obs, _, done, _ = env.step(act)
            steps += 1
            if done:
                success = True
                break
        env.close()
        result.update({"status": "ok", "success": success, "num_steps": steps, "error": ""})
    except Exception as exc:
        result["error"] = repr(exc)
    return result


@app.local_entrypoint()
def main(task_id: int = 0) -> None:
    print(f"X-VLA spike: task_id={task_id} center=({CENTER_X}, {CENTER_Y})")
    result = run_spike.remote(task_id, CENTER_X, CENTER_Y)
    out = LOCAL_ROOT / "xvla_spike_result.json"
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print(f"Wrote {out}")
