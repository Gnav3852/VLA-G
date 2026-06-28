#!/usr/bin/env python3
"""
OpenVLA-OFT runnability spike on Modal/LIBERO.

  ./.venv/bin/modal run modal_oft_spike.py

Writes oft_spike_result.json locally with status ok/error.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import modal

APP_NAME = "vla-oft-spike"
OFT_REPO = "https://github.com/moojink/openvla-oft.git"
CHECKPOINT = "moojink/openvla-7b-oft-finetuned-libero-spatial"
GPU_TYPE = "A10"
LOCAL_ROOT = Path(__file__).resolve().parent

TASK_SUITE = "libero_spatial"
TARGET_OBJECT = "akita_black_bowl_1"
CENTER_X = -0.06
CENTER_Y = 0.20

app = modal.App(APP_NAME)
weights_volume = modal.Volume.from_name("model-weights-oft", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install(
        "build-essential", "git", "libglib2.0-0", "libegl1", "libegl1-mesa",
        "libgl1", "libgl1-mesa-glx", "libgles2", "libglvnd-dev", "libglvnd0",
        "libosmesa6-dev", "mesa-utils", "ninja-build", "xvfb",
    )
    .env(
        {
            "MUJOCO_GL": "egl",
            "PYOPENGL_PLATFORM": "egl",
            "NVIDIA_DRIVER_CAPABILITIES": "all",
            "PYTHONPATH": "/opt/openvla-oft:/opt/LIBERO",
            "HF_HOME": "/weights/hf-cache",
            "TRANSFORMERS_CACHE": "/weights/hf-cache",
        }
    )
    .pip_install("packaging", "wheel", "setuptools", "ninja")
    .pip_install(
        "torch==2.2.0", "torchvision==0.17.0", "torchaudio==2.2.0",
        index_url="https://download.pytorch.org/whl/cu121",
    )
    .pip_install(
        "accelerate>=0.25.0", "draccus==0.8.0", "einops", "huggingface_hub<1.0",
        "json-numpy", "matplotlib", "numpy<2", "peft==0.11.1", "protobuf==3.20.3",
        "rich", "safetensors", "sentencepiece==0.1.99", "tensorflow==2.15.0",
        "timm==0.9.10", "tokenizers==0.19.1", "transformers==4.40.1", "wandb",
        "diffusers",
    )
    .pip_install(
        "PyOpenGL", "bddl", "cloudpickle", "easydict", "gym", "imageio[ffmpeg]",
        "mujoco==2.3.7", "robosuite==1.4.1",
    )
    .pip_install("numpy<2", "opencv-python==4.9.0.80")
    .run_commands(
        f"git clone --depth 1 {OFT_REPO} /opt/openvla-oft",
        "pip install -e /opt/openvla-oft --no-deps",
        "pip install git+https://github.com/moojink/dlimp_openvla",
        "pip install tensorflow_datasets==4.9.3 tensorflow_graphics==2021.12.3 tensorflow_metadata==1.14.0 jsonlines",
        "git clone --depth 1 https://github.com/Lifelong-Robot-Learning/LIBERO.git /opt/LIBERO",
        "pip install -e /opt/LIBERO --no-deps",
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


@app.function(image=image, gpu=GPU_TYPE, timeout=60 * 30, volumes={"/weights": weights_volume})
def run_spike(task_id: int, grid_x: float, grid_y: float) -> dict[str, Any]:
    import numpy as np
    import torch
    from libero.libero import benchmark
    from experiments.robot.openvla_utils import get_processor, get_vla, get_vla_action
    from experiments.robot.robot_utils import (
        get_libero_dummy_action,
        get_libero_env,
        get_libero_image,
        invert_gripper_action,
        normalize_gripper_action,
        set_seed_everywhere,
    )
    from prismatic.vla.constants import NUM_ACTIONS_CHUNK

    class _Cfg:
        pretrained_checkpoint = CHECKPOINT
        use_l1_regression = True
        use_diffusion = False
        use_film = False
        num_images_in_input = 2
        use_proprio = True
        load_in_8bit = False
        load_in_4bit = False
        center_crop = True
        num_open_loop_steps = NUM_ACTIONS_CHUNK
        unnorm_key = "libero_spatial_no_noops"

    result: dict[str, Any] = {
        "model": "openvla_oft",
        "checkpoint": CHECKPOINT,
        "status": "error",
        "success": False,
        "num_steps": 0,
        "error": "",
    }
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable")
        set_seed_everywhere(7)
        cfg = _Cfg()
        vla = get_vla(cfg)
        processor = get_processor(cfg)
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
            img = get_libero_image(obs, 224)
            state = np.concatenate(
                (obs["robot0_eef_pos"], obs["robot0_eef_quat"][:3], obs["robot0_gripper_qpos"])
            )
            observation = {"full_image": img, "state": state}
            action = get_vla_action(cfg, vla, processor, observation, task_description)
            action = normalize_gripper_action(action, binarize=True)
            action = invert_gripper_action(action)
            obs, _, done, _ = env.step(action.tolist())
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
    gx, gy = CENTER_X, CENTER_Y
    print(f"OFT spike: task_id={task_id} center=({gx}, {gy})")
    result = run_spike.remote(task_id, gx, gy)
    out = LOCAL_ROOT / "oft_spike_result.json"
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print(f"Wrote {out}")
