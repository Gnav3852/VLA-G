#!/usr/bin/env python3
"""
Live PatchMask action-shift probe on LIBERO-Spatial (OpenVLA + Pi0.5).

Paired inference at each timestep: same physical state, unmasked vs masked image.
Behavioral score = mean per-step L2(action_unmasked, action_masked) — no VLA-Trace ΔSR.

  modal run modal_patchmask_probe.py --use-proxy=false --test
  modal run modal_patchmask_probe.py --use-proxy=false
  modal run modal_patchmask_probe.py --use-proxy=true   # circular proxy (audit only)
"""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

import modal
import numpy as np

APP_NAME = "vla-patchmask-probe"
LOCAL_ROOT = Path(__file__).resolve().parent
PROBE_DIR = "/opt/probe"

TASK_SUITE = "libero_spatial"
MODEL_SEED = 7
ENV_SEED = 0
NUM_STEPS_WAIT = 10
POSITION_TOLERANCE_M = 0.015

OPENVLA_MODEL_ID = "openvla/openvla-7b-finetuned-libero-spatial"
OPENVLA_WEIGHTS = "/weights/openvla-7b-finetuned-libero-spatial"
PI05_CONFIG = "pi05_libero"
PI05_CHECKPOINT = f"/weights/{PI05_CONFIG}"
OPENPI_REV = "981483dca0fd9acba698fea00aa6e52d56a66c58"

OFT_REPO = "https://github.com/moojink/openvla-oft.git"
OFT_CHECKPOINT = "moojink/openvla-7b-oft-finetuned-libero-spatial"
OFT_WEIGHTS = "/weights/hf-cache"

XVLA_CHECKPOINT = "lerobot/xvla-libero"
XVLA_WEIGHTS = "/weights/hf-cache"

PROBE_CONDITIONS = [
    ("mask_target", "black", "target_black"),
    ("mask_target", "background_fill", "target_bg"),
    ("mask_gripper", "black", "gripper_black"),
    ("mask_robot", "black", "robot_black"),
    ("mask_background", "black", "background_black"),
]

SMOKE_CONDITIONS = [
    ("mask_target", "black", "target_black"),
    ("mask_background", "black", "background_black"),
]

app = modal.App(APP_NAME)
openvla_weights = modal.Volume.from_name("model-weights-openvla", create_if_missing=True)
pi05_weights = modal.Volume.from_name("model-weights-pi05", create_if_missing=True)
oft_weights = modal.Volume.from_name("model-weights-oft", create_if_missing=True)
xvla_weights = modal.Volume.from_name("model-weights-xvla", create_if_missing=True)


def _add_probe_sources(image: modal.Image) -> modal.Image:
    for name in ("patchmask_probe.py", "libero_patchmask.py", "patchmask_live_metrics.py"):
        image = image.add_local_file(LOCAL_ROOT / name, f"{PROBE_DIR}/{name}")
    return image


openvla_image = (
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
            "HF_HOME": "/weights/hf-cache",
            "TRANSFORMERS_CACHE": "/weights/hf-cache",
            "TORCH_HOME": "/weights/torch-cache",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "PYTHONPATH": f"{PROBE_DIR}:/opt/openvla:/opt/LIBERO",
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
        "pillow",
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
        "opencv-python==4.9.0.80",
    )
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
openvla_image = _add_probe_sources(openvla_image)

pi05_image = (
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
            "PYTHONPATH": f"{PROBE_DIR}:/opt/openpi/src:/opt/LIBERO",
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
            "XLA_PYTHON_CLIENT_ALLOCATOR": "platform",
        }
    )
    .pip_install("packaging", "wheel", "setuptools", "ninja", "hatchling", "pillow")
    .pip_install(
        "jax[cuda12]==0.5.3",
        "flax==0.10.2",
        "orbax-checkpoint==0.11.13",
        "chex",
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
        "numpy<2",
        "numpydantic>=1.6.6",
        "opencv-python==4.9.0.80",
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
    .pip_install("numpy>=1.22.4,<2.0.0")
    .run_commands(
        "mkdir -p /root/.libero",
        "python - <<'PY'\n"
        "from pathlib import Path\n"
        "import yaml\n"
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
pi05_image = _add_probe_sources(pi05_image)

oft_image = (
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
            "HF_HOME": "/weights/hf-cache",
            "TRANSFORMERS_CACHE": "/weights/hf-cache",
            "PYTHONPATH": f"{PROBE_DIR}:/opt/openvla-oft:/opt/LIBERO",
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
        "diffusers==0.25.1", "pillow",
    )
    .pip_install(
        "PyOpenGL", "bddl", "cloudpickle", "easydict", "gym", "imageio[ffmpeg]",
        "mujoco==2.3.7", "robosuite==1.4.1", "opencv-python==4.9.0.80",
    )
    .run_commands(
        f"git clone --depth 1 {OFT_REPO} /opt/openvla-oft",
        "pip install -e /opt/openvla-oft --no-deps",
        "pip install git+https://github.com/moojink/dlimp_openvla",
        "pip install tensorflow_datasets==4.9.3 tensorflow_graphics==2021.12.3 tensorflow_metadata==1.14.0 jsonlines",
        "git clone --depth 1 https://github.com/Lifelong-Robot-Learning/LIBERO.git /opt/LIBERO",
        "pip install -e /opt/LIBERO --no-deps",
    )
    .run_commands(
        "mkdir -p /root/.libero",
        "python - <<'PY'\n"
        "from pathlib import Path\n"
        "import yaml\n"
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
oft_image = _add_probe_sources(oft_image)

xvla_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(
        "build-essential", "git", "libglib2.0-0", "libegl1", "libegl1-mesa",
        "libgl1", "libgl1-mesa-glx", "libgles2", "libglvnd-dev", "libglvnd0",
        "libosmesa6-dev", "mesa-utils", "xvfb",
    )
    .env(
        {
            "MUJOCO_GL": "egl",
            "PYOPENGL_PLATFORM": "egl",
            "NVIDIA_DRIVER_CAPABILITIES": "all",
            "HF_HOME": "/weights/hf-cache",
            "PYTHONPATH": f"{PROBE_DIR}:/opt/LIBERO",
        }
    )
    .pip_install("packaging", "wheel", "setuptools")
    .pip_install(
        "torch", "torchvision", "transformers", "accelerate",
        "numpy<2", "pillow",
    )
    .pip_install("lerobot")
    .run_commands(
        "git clone --depth 1 https://github.com/Lifelong-Robot-Learning/LIBERO.git /opt/LIBERO",
        "pip install -e /opt/LIBERO --no-deps",
        "pip install mujoco==2.3.7 robosuite==1.4.1 opencv-python==4.9.0.80",
    )
    .pip_install("numpy>=1.22.4,<2.0.0")
    .run_commands(
        "mkdir -p /root/.libero",
        "python - <<'PY'\n"
        "from pathlib import Path\n"
        "import yaml\n"
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
xvla_image = _add_probe_sources(xvla_image)


def _shallow_obs_copy(obs: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, val in obs.items():
        out[key] = val.copy() if hasattr(val, "copy") else val
    return out


@app.function(image=openvla_image, volumes={"/weights": openvla_weights}, gpu="A10", timeout=3600, retries=1)
def probe_openvla_live_batch(jobs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    import gc
    import os
    import sys
    from dataclasses import dataclass
    from io import BytesIO
    from pathlib import Path

    import numpy as np
    import torch
    from PIL import Image

    sys.path.insert(0, PROBE_DIR)
    sys.path.insert(0, "/opt/LIBERO")
    sys.path.insert(0, "/opt/openvla")
    os.chdir("/opt/openvla")
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

    from libero.libero import benchmark
    from libero_patchmask import (
        apply_image_mask_to_obs_inplace,
        compute_mask_verification,
        detect_instance_seg_keys,
        make_segmentation_env,
    )
    from patchmask_live_metrics import aggregate_shifts, per_step_shifts

    from experiments.robot.libero.libero_utils import get_libero_dummy_action, get_libero_image, quat2axisangle
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
        import json as json_mod

        from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
        from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
        from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
        from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor

        def get_vla_sdpa(cfg):
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
                    vla.norm_stats = json_mod.load(f)
            return vla

        openvla_utils.get_vla = get_vla_sdpa
        robot_utils.get_vla = get_vla_sdpa

    @dataclass
    class ModelCfg:
        model_family: str = "openvla"
        pretrained_checkpoint: str = OPENVLA_WEIGHTS
        load_in_8bit: bool = False
        load_in_4bit: bool = False
        center_crop: bool = True
        task_suite_name: str = TASK_SUITE
        unnorm_key: str = TASK_SUITE
        num_steps_wait: int = NUM_STEPS_WAIT

    def patch_target_object_xy(env, state, obj_name: str, x: float, y: float):
        patched_state = np.asarray(state, dtype=np.float64).copy()
        obj = env.env.get_object(obj_name)
        joint_name = obj.joints[-1]
        addr = env.sim.model.get_joint_qpos_addr(joint_name)
        start, end = (addr if isinstance(addr, tuple) else (addr, addr + 1))
        qpos_start = 1 + start
        patched_state[qpos_start] = x
        patched_state[qpos_start + 1] = y
        return patched_state, float(patched_state[qpos_start + 2])

    def openvla_action(obs, task_description, model, processor, model_cfg, resize_size):
        with torch.inference_mode():
            img = get_libero_image(obs, resize_size)
            observation = {
                "full_image": img,
                "state": np.concatenate(
                    (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
                ),
            }
            action = get_action(model_cfg, model, observation, task_description, processor=processor)
            action = normalize_gripper_action(action, binarize=True)
            return invert_gripper_action(action)

    def run_one(job: dict[str, Any]) -> dict[str, Any]:
        variant = job["variant"]
        mode = job["mode"]
        setting_key = job["setting_key"]
        task_id = job["task_id"]
        base_idx = job["base_idx"]
        center_x = job["center_x"]
        center_y = job["center_y"]
        target_object = job["target_object"]
        max_probe_steps = job["max_probe_steps"]
        save_frame = job.get("save_frame", False)

        result: dict[str, Any] = {
            "model": "openvla",
            "variant": variant,
            "mode": mode,
            "setting": setting_key,
            "suite": TASK_SUITE,
            "task_id": task_id,
            "base_idx": base_idx,
            "status": "error",
            "source": "live_rollout",
        }
        env = None
        try:
            task_suite = benchmark.get_benchmark_dict()[TASK_SUITE]()
            task = task_suite.get_task(task_id)
            initial_states = task_suite.get_task_init_states(task_id)
            episode_idx = int(base_idx) % len(initial_states)
            env, task_description = make_segmentation_env(task, seed=ENV_SEED)

            env.reset()
            patched_state, _ = patch_target_object_xy(
                env, initial_states[episode_idx], target_object, center_x, center_y
            )
            obs = env.set_init_state(patched_state)

            seg_keys: tuple[str, str] | None = None
            actions_clean: list[np.ndarray] = []
            actions_masked: list[np.ndarray] = []
            frame_b64: str | None = None
            mask_verification: dict[str, Any] | None = None

            t = 0
            while t < NUM_STEPS_WAIT + max_probe_steps:
                if t < NUM_STEPS_WAIT:
                    obs, _, _, _ = env.step(get_libero_dummy_action("openvla"))
                    t += 1
                    continue

                if seg_keys is None:
                    seg_keys = detect_instance_seg_keys(obs)
                    mask_verification = compute_mask_verification(
                        obs,
                        env,
                        variant=variant,
                        seg_keys=seg_keys,
                        target_object=target_object,
                    )

                a_clean = openvla_action(obs, task_description, model, processor, model_cfg, resize_size)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                obs_m = _shallow_obs_copy(obs)
                apply_image_mask_to_obs_inplace(obs_m, env, variant=variant, mode=mode, seg_keys=seg_keys)
                a_masked = openvla_action(obs_m, task_description, model, processor, model_cfg, resize_size)

                if save_frame and frame_b64 is None:
                    img = np.ascontiguousarray(obs_m["agentview_image"][::-1, ::-1])
                    buf = BytesIO()
                    Image.fromarray(img).save(buf, format="PNG")
                    frame_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

                actions_clean.append(np.asarray(a_clean, dtype=np.float64))
                actions_masked.append(np.asarray(a_masked, dtype=np.float64))
                obs, _, done, _ = env.step(a_clean.tolist() if hasattr(a_clean, "tolist") else list(a_clean))
                t += 1
                if done:
                    break

            raw_list, norm_list = per_step_shifts(actions_clean, actions_masked)
            metrics = aggregate_shifts(raw_list, norm_list)
            result.update(
                {
                    "status": "ok",
                    **metrics,
                    "action_sensitivity": metrics["raw_l2_shift"],
                    "n_rollouts": 1,
                    "frame_png_b64": frame_b64,
                    "mask_verification": mask_verification,
                }
            )
        except Exception as exc:
            result["error"] = repr(exc)
        finally:
            if env is not None:
                env.close()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return result

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    patch_openvla_loader_for_sdpa()
    set_seed_everywhere(MODEL_SEED)
    model_cfg = ModelCfg()
    model = get_model(model_cfg)
    processor = get_processor(model_cfg)
    resize_size = get_image_resize_size(model_cfg)

    return [run_one(job) for job in jobs]


@app.function(image=pi05_image, volumes={"/weights": pi05_weights}, gpu="A100", timeout=3600, retries=1)
def probe_pi05_live_batch(jobs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    import gc
    import math
    import os
    import sys
    from io import BytesIO

    import numpy as np
    import torch
    from PIL import Image

    sys.path.insert(0, PROBE_DIR)
    sys.path.insert(0, "/opt/openpi/src")
    sys.path.insert(0, "/opt/LIBERO")
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

    import jax

    jax.config.update("jax_platform_name", "gpu")

    from libero.libero import benchmark
    from libero_patchmask import (
        apply_image_mask_to_obs_inplace,
        compute_mask_verification,
        detect_instance_seg_keys,
        make_segmentation_env,
    )
    from openpi.policies import policy_config
    from openpi.training import config as openpi_config
    from openpi_client import image_tools
    from patchmask_live_metrics import aggregate_shifts, per_step_shifts

    _orig_torch_load = torch.load

    def _torch_load_compat(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return _orig_torch_load(*args, **kwargs)

    torch.load = _torch_load_compat

    LIBERO_DUMMY = [0.0] * 6 + [-1.0]
    IMAGE_SIZE = 224

    def quat2axisangle(quat: np.ndarray) -> np.ndarray:
        q = np.asarray(quat, dtype=np.float64)
        q[3] = float(np.clip(q[3], -1.0, 1.0))
        den = np.sqrt(max(0.0, 1.0 - q[3] * q[3]))
        if math.isclose(den, 0.0):
            return np.zeros(3, dtype=np.float64)
        return (q[:3] * 2.0 * math.acos(q[3])) / den

    def build_pi_obs(obs, task_description: str) -> dict[str, Any]:
        img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
        wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
        img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, IMAGE_SIZE, IMAGE_SIZE))
        wrist = image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist, IMAGE_SIZE, IMAGE_SIZE))
        state = np.concatenate(
            [obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"]],
            dtype=np.float64,
        )
        return {
            "observation/image": img,
            "observation/wrist_image": wrist,
            "observation/state": state,
            "prompt": str(task_description),
        }

    def patch_target_object_xy(env, state, obj_name: str, x: float, y: float):
        patched_state = np.asarray(state, dtype=np.float64).copy()
        obj = env.env.get_object(obj_name)
        joint_name = obj.joints[-1]
        addr = env.sim.model.get_joint_qpos_addr(joint_name)
        start, end = (addr if isinstance(addr, tuple) else (addr, addr + 1))
        qpos_start = 1 + start
        patched_state[qpos_start] = x
        patched_state[qpos_start + 1] = y
        return patched_state, float(patched_state[qpos_start + 2])

    cfg = openpi_config.get_config(PI05_CONFIG)
    policy = policy_config.create_trained_policy(cfg, PI05_CHECKPOINT)

    def run_one(job: dict[str, Any]) -> dict[str, Any]:
        variant = job["variant"]
        mode = job["mode"]
        setting_key = job["setting_key"]
        task_id = job["task_id"]
        base_idx = job["base_idx"]
        center_x = job["center_x"]
        center_y = job["center_y"]
        target_object = job["target_object"]
        max_probe_steps = job["max_probe_steps"]
        save_frame = job.get("save_frame", False)

        result: dict[str, Any] = {
            "model": "pi05",
            "variant": variant,
            "mode": mode,
            "setting": setting_key,
            "suite": TASK_SUITE,
            "task_id": task_id,
            "base_idx": base_idx,
            "status": "error",
            "source": "live_rollout",
        }
        env = None
        try:
            task_suite = benchmark.get_benchmark_dict()[TASK_SUITE]()
            task = task_suite.get_task(task_id)
            initial_states = task_suite.get_task_init_states(task_id)
            episode_idx = int(base_idx) % len(initial_states)
            env, task_description = make_segmentation_env(task, seed=ENV_SEED)

            env.reset()
            patched_state, _ = patch_target_object_xy(
                env, initial_states[episode_idx], target_object, center_x, center_y
            )
            obs = env.set_init_state(patched_state)

            seg_keys: tuple[str, str] | None = None
            actions_clean: list[np.ndarray] = []
            actions_masked: list[np.ndarray] = []
            frame_b64: str | None = None
            mask_verification: dict[str, Any] | None = None

            t = 0
            while t < NUM_STEPS_WAIT + max_probe_steps:
                if t < NUM_STEPS_WAIT:
                    obs, _, _, _ = env.step(LIBERO_DUMMY)
                    t += 1
                    continue

                if seg_keys is None:
                    seg_keys = detect_instance_seg_keys(obs)
                    mask_verification = compute_mask_verification(
                        obs,
                        env,
                        variant=variant,
                        seg_keys=seg_keys,
                        target_object=target_object,
                    )

                out_clean = policy.infer(build_pi_obs(obs, task_description))
                a_clean = np.asarray(out_clean["actions"][0], dtype=np.float64)

                obs_m = _shallow_obs_copy(obs)
                apply_image_mask_to_obs_inplace(obs_m, env, variant=variant, mode=mode, seg_keys=seg_keys)
                out_masked = policy.infer(build_pi_obs(obs_m, task_description))
                a_masked = np.asarray(out_masked["actions"][0], dtype=np.float64)

                if save_frame and frame_b64 is None:
                    img = np.ascontiguousarray(obs_m["agentview_image"][::-1, ::-1])
                    buf = BytesIO()
                    Image.fromarray(img).save(buf, format="PNG")
                    frame_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

                actions_clean.append(a_clean)
                actions_masked.append(a_masked)
                obs, _, done, _ = env.step(a_clean.tolist())
                t += 1
                if done:
                    break

            raw_list, norm_list = per_step_shifts(actions_clean, actions_masked)
            metrics = aggregate_shifts(raw_list, norm_list)
            result.update(
                {
                    "status": "ok",
                    **metrics,
                    "action_sensitivity": metrics["raw_l2_shift"],
                    "n_rollouts": 1,
                    "frame_png_b64": frame_b64,
                    "mask_verification": mask_verification,
                }
            )
        except Exception as exc:
            result["error"] = repr(exc)
        finally:
            if env is not None:
                env.close()
            gc.collect()
        return result

    return [run_one(job) for job in jobs]


@app.function(image=oft_image, volumes={"/weights": oft_weights}, gpu="A10", timeout=3600, retries=1)
def probe_oft_live_batch(jobs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    import gc
    import os
    import sys
    from dataclasses import dataclass
    from io import BytesIO

    import numpy as np
    import torch
    from PIL import Image

    sys.path.insert(0, PROBE_DIR)
    sys.path.insert(0, "/opt/LIBERO")
    sys.path.insert(0, "/opt/openvla-oft")
    os.chdir("/opt/openvla-oft")
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

    from libero.libero import benchmark
    from libero_patchmask import (
        apply_image_mask_to_obs_inplace,
        compute_mask_verification,
        detect_instance_seg_keys,
        make_segmentation_env,
    )
    from patchmask_live_metrics import aggregate_shifts, per_step_shifts

    from experiments.robot.libero.libero_utils import (
        get_libero_dummy_action,
        get_libero_image,
        get_libero_wrist_image,
        quat2axisangle,
    )
    import experiments.robot.openvla_utils as openvla_utils
    import experiments.robot.robot_utils as robot_utils
    from experiments.robot.openvla_utils import get_processor
    from experiments.robot.robot_utils import (
        invert_gripper_action,
        normalize_gripper_action,
        set_seed_everywhere,
    )
    from prismatic.vla.constants import NUM_ACTIONS_CHUNK, PROPRIO_DIM

    @dataclass
    class OFTCfg:
        model_family: str = "openvla"
        pretrained_checkpoint: str = OFT_CHECKPOINT
        use_l1_regression: bool = True
        use_diffusion: bool = False
        use_film: bool = False
        num_images_in_input: int = 2
        use_proprio: bool = True
        load_in_8bit: bool = False
        load_in_4bit: bool = False
        center_crop: bool = True
        num_open_loop_steps: int = NUM_ACTIONS_CHUNK
        unnorm_key: str = "libero_spatial_no_noops"
        task_suite_name: str = TASK_SUITE

    def patch_openvla_loader_for_sdpa():
        import json as json_mod
        from pathlib import Path as _Path
        from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
        from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
        from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
        from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor

        def get_vla_sdpa(cfg):
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
            stats_path = _Path(cfg.pretrained_checkpoint) / "dataset_statistics.json"
            if stats_path.is_file():
                with stats_path.open() as f:
                    vla.norm_stats = json_mod.load(f)
            return vla

        openvla_utils.get_vla = get_vla_sdpa
        robot_utils.get_vla = get_vla_sdpa

    def patch_target_object_xy(env, state, obj_name: str, x: float, y: float):
        patched_state = np.asarray(state, dtype=np.float64).copy()
        obj = env.env.get_object(obj_name)
        joint_name = obj.joints[-1]
        addr = env.sim.model.get_joint_qpos_addr(joint_name)
        start, end = (addr if isinstance(addr, tuple) else (addr, addr + 1))
        qpos_start = 1 + start
        patched_state[qpos_start] = x
        patched_state[qpos_start + 1] = y
        return patched_state, float(patched_state[qpos_start + 2])

    def oft_action(obs, task_description, vla, processor, cfg, action_head, proprio_projector):
        with torch.inference_mode():
            full_img = get_libero_image(obs)
            wrist_img = get_libero_wrist_image(obs)
            state = np.concatenate(
                (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
            )
            observation = {
                "full_image": full_img,
                "wrist_image": wrist_img,
                "state": state,
            }
            actions = openvla_utils.get_vla_action(
                cfg, vla, processor, observation, task_description,
                action_head=action_head,
                proprio_projector=proprio_projector,
            )
            action = actions[0] if hasattr(actions, "__len__") and len(actions) > 0 else actions
            action = normalize_gripper_action(action, binarize=True)
            return invert_gripper_action(action)

    def run_one(job: dict[str, Any]) -> dict[str, Any]:
        variant = job["variant"]
        mode = job["mode"]
        setting_key = job["setting_key"]
        task_id = job["task_id"]
        base_idx = job["base_idx"]
        center_x = job["center_x"]
        center_y = job["center_y"]
        target_object = job["target_object"]
        max_probe_steps = job["max_probe_steps"]
        save_frame = job.get("save_frame", False)

        result: dict[str, Any] = {
            "model": "openvla_oft",
            "variant": variant,
            "mode": mode,
            "setting": setting_key,
            "suite": TASK_SUITE,
            "task_id": task_id,
            "base_idx": base_idx,
            "status": "error",
            "source": "live_rollout",
        }
        env = None
        try:
            task_suite = benchmark.get_benchmark_dict()[TASK_SUITE]()
            task = task_suite.get_task(task_id)
            initial_states = task_suite.get_task_init_states(task_id)
            episode_idx = int(base_idx) % len(initial_states)
            env, task_description = make_segmentation_env(task, seed=ENV_SEED)

            env.reset()
            patched_state, _ = patch_target_object_xy(
                env, initial_states[episode_idx], target_object, center_x, center_y
            )
            obs = env.set_init_state(patched_state)

            seg_keys: tuple[str, str] | None = None
            actions_clean: list[np.ndarray] = []
            actions_masked: list[np.ndarray] = []
            frame_b64: str | None = None
            mask_verification: dict[str, Any] | None = None

            t = 0
            while t < NUM_STEPS_WAIT + max_probe_steps:
                if t < NUM_STEPS_WAIT:
                    obs, _, _, _ = env.step(get_libero_dummy_action("openvla"))
                    t += 1
                    continue

                if seg_keys is None:
                    seg_keys = detect_instance_seg_keys(obs)
                    mask_verification = compute_mask_verification(
                        obs, env, variant=variant, seg_keys=seg_keys, target_object=target_object,
                    )

                a_clean = oft_action(obs, task_description, vla, processor, cfg, action_head, proprio_projector)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                obs_m = _shallow_obs_copy(obs)
                apply_image_mask_to_obs_inplace(obs_m, env, variant=variant, mode=mode, seg_keys=seg_keys)
                a_masked = oft_action(obs_m, task_description, vla, processor, cfg, action_head, proprio_projector)

                if save_frame and frame_b64 is None:
                    img = np.ascontiguousarray(obs_m["agentview_image"][::-1, ::-1])
                    buf = BytesIO()
                    Image.fromarray(img).save(buf, format="PNG")
                    frame_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

                actions_clean.append(np.asarray(a_clean, dtype=np.float64))
                actions_masked.append(np.asarray(a_masked, dtype=np.float64))
                obs, _, done, _ = env.step(a_clean.tolist() if hasattr(a_clean, "tolist") else list(a_clean))
                t += 1
                if done:
                    break

            raw_list, norm_list = per_step_shifts(actions_clean, actions_masked)
            metrics = aggregate_shifts(raw_list, norm_list)
            result.update(
                {
                    "status": "ok",
                    **metrics,
                    "action_sensitivity": metrics["raw_l2_shift"],
                    "n_rollouts": 1,
                    "frame_png_b64": frame_b64,
                    "mask_verification": mask_verification,
                }
            )
        except Exception as exc:
            result["error"] = repr(exc)
        finally:
            if env is not None:
                env.close()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return result

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    patch_openvla_loader_for_sdpa()
    set_seed_everywhere(MODEL_SEED)
    cfg = OFTCfg()
    vla = openvla_utils.get_vla(cfg)
    processor = get_processor(cfg)
    action_head = openvla_utils.get_action_head(cfg, llm_dim=vla.llm_dim)
    proprio_projector = openvla_utils.get_proprio_projector(cfg, llm_dim=vla.llm_dim, proprio_dim=PROPRIO_DIM)

    return [run_one(job) for job in jobs]


@app.function(image=xvla_image, volumes={"/weights": xvla_weights}, gpu="A10", timeout=3600, retries=1)
def probe_xvla_live_batch(jobs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    import gc
    import os
    import sys
    from io import BytesIO

    import numpy as np
    import torch
    from PIL import Image

    sys.path.insert(0, PROBE_DIR)
    sys.path.insert(0, "/opt/LIBERO")
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

    from libero.libero import benchmark
    from libero_patchmask import (
        apply_image_mask_to_obs_inplace,
        compute_mask_verification,
        detect_instance_seg_keys,
        make_segmentation_env,
    )
    from patchmask_live_metrics import aggregate_shifts, per_step_shifts

    LIBERO_DUMMY = [0.0] * 6 + [-1.0]
    IMAGE_SIZE = 224

    def patch_target_object_xy(env, state, obj_name: str, x: float, y: float):
        patched_state = np.asarray(state, dtype=np.float64).copy()
        obj = env.env.get_object(obj_name)
        joint_name = obj.joints[-1]
        addr = env.sim.model.get_joint_qpos_addr(joint_name)
        start, end = (addr if isinstance(addr, tuple) else (addr, addr + 1))
        qpos_start = 1 + start
        patched_state[qpos_start] = x
        patched_state[qpos_start + 1] = y
        return patched_state, float(patched_state[qpos_start + 2])

    def build_xvla_obs(obs, task_description: str) -> dict[str, Any]:
        img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
        img_pil = Image.fromarray(img).resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
        img_t = torch.from_numpy(np.array(img_pil)).permute(2, 0, 1).float() / 255.0
        state = np.concatenate(
            [obs["robot0_eef_pos"], obs["robot0_eef_quat"], obs["robot0_gripper_qpos"]],
            dtype=np.float64,
        )
        return {
            "observation.images.image": img_t.unsqueeze(0).to(device),
            "observation.state": torch.from_numpy(state).float().unsqueeze(0).to(device),
            "task": [task_description],
        }

    def run_one(job: dict[str, Any]) -> dict[str, Any]:
        variant = job["variant"]
        mode = job["mode"]
        setting_key = job["setting_key"]
        task_id = job["task_id"]
        base_idx = job["base_idx"]
        center_x = job["center_x"]
        center_y = job["center_y"]
        target_object = job["target_object"]
        max_probe_steps = job["max_probe_steps"]
        save_frame = job.get("save_frame", False)

        result: dict[str, Any] = {
            "model": "x_vla",
            "variant": variant,
            "mode": mode,
            "setting": setting_key,
            "suite": TASK_SUITE,
            "task_id": task_id,
            "base_idx": base_idx,
            "status": "error",
            "source": "live_rollout",
        }
        env = None
        try:
            task_suite = benchmark.get_benchmark_dict()[TASK_SUITE]()
            task = task_suite.get_task(task_id)
            initial_states = task_suite.get_task_init_states(task_id)
            episode_idx = int(base_idx) % len(initial_states)
            env, task_description = make_segmentation_env(task, seed=ENV_SEED)

            env.reset()
            patched_state, _ = patch_target_object_xy(
                env, initial_states[episode_idx], target_object, center_x, center_y
            )
            obs = env.set_init_state(patched_state)

            seg_keys: tuple[str, str] | None = None
            actions_clean: list[np.ndarray] = []
            actions_masked: list[np.ndarray] = []
            frame_b64: str | None = None
            mask_verification: dict[str, Any] | None = None

            t = 0
            while t < NUM_STEPS_WAIT + max_probe_steps:
                if t < NUM_STEPS_WAIT:
                    obs, _, _, _ = env.step(LIBERO_DUMMY)
                    t += 1
                    continue

                if seg_keys is None:
                    seg_keys = detect_instance_seg_keys(obs)
                    mask_verification = compute_mask_verification(
                        obs, env, variant=variant, seg_keys=seg_keys, target_object=target_object,
                    )

                with torch.no_grad():
                    a_clean = policy.select_action(build_xvla_obs(obs, task_description))
                a_clean = a_clean[0].detach().cpu().numpy()

                obs_m = _shallow_obs_copy(obs)
                apply_image_mask_to_obs_inplace(obs_m, env, variant=variant, mode=mode, seg_keys=seg_keys)
                with torch.no_grad():
                    a_masked = policy.select_action(build_xvla_obs(obs_m, task_description))
                a_masked = a_masked[0].detach().cpu().numpy()

                if save_frame and frame_b64 is None:
                    img = np.ascontiguousarray(obs_m["agentview_image"][::-1, ::-1])
                    buf = BytesIO()
                    Image.fromarray(img).save(buf, format="PNG")
                    frame_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

                actions_clean.append(np.asarray(a_clean, dtype=np.float64))
                actions_masked.append(np.asarray(a_masked, dtype=np.float64))
                obs, _, done, _ = env.step(a_clean.tolist())
                t += 1
                if done:
                    break

            raw_list, norm_list = per_step_shifts(actions_clean, actions_masked)
            metrics = aggregate_shifts(raw_list, norm_list)
            result.update(
                {
                    "status": "ok",
                    **metrics,
                    "action_sensitivity": metrics["raw_l2_shift"],
                    "n_rollouts": 1,
                    "frame_png_b64": frame_b64,
                    "mask_verification": mask_verification,
                }
            )
        except Exception as exc:
            result["error"] = repr(exc)
        finally:
            if env is not None:
                env.close()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return result

    from lerobot.common.policies.xvla.modeling_xvla import XVLAPolicy

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = XVLAPolicy.from_pretrained(XVLA_CHECKPOINT).to(device).eval()

    return [run_one(job) for job in jobs]


@app.function(image=openvla_image, gpu="A10", timeout=900, retries=1)
def verify_masks_env_only(jobs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Env-only mask verification (no VLA load). Saves masked frames + coverage stats."""
    import gc
    import os
    import sys
    from io import BytesIO

    import numpy as np
    from PIL import Image

    sys.path.insert(0, PROBE_DIR)
    sys.path.insert(0, "/opt/LIBERO")
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

    from libero.libero import benchmark
    from libero_patchmask import (
        apply_image_mask_to_obs_inplace,
        build_robot_target_masks,
        compute_mask_verification,
        detect_instance_seg_keys,
        make_segmentation_env,
    )

    from experiments.robot.libero.libero_utils import get_libero_dummy_action

    def patch_target_object_xy(env, state, obj_name: str, x: float, y: float):
        patched_state = np.asarray(state, dtype=np.float64).copy()
        obj = env.env.get_object(obj_name)
        joint_name = obj.joints[-1]
        addr = env.sim.model.get_joint_qpos_addr(joint_name)
        start, end = (addr if isinstance(addr, tuple) else (addr, addr + 1))
        qpos_start = 1 + start
        patched_state[qpos_start] = x
        patched_state[qpos_start + 1] = y
        return patched_state

    def run_one(job: dict[str, Any]) -> dict[str, Any]:
        variant = job["variant"]
        mode = job["mode"]
        task_id = job["task_id"]
        base_idx = job["base_idx"]
        center_x = job["center_x"]
        center_y = job["center_y"]
        target_object = job["target_object"]
        result: dict[str, Any] = {
            "model": job.get("model", "env_only"),
            "variant": variant,
            "mode": mode,
            "task_id": task_id,
            "base_idx": base_idx,
            "status": "error",
        }
        env = None
        try:
            task_suite = benchmark.get_benchmark_dict()[TASK_SUITE]()
            task = task_suite.get_task(task_id)
            initial_states = task_suite.get_task_init_states(task_id)
            episode_idx = int(base_idx) % len(initial_states)
            env, _ = make_segmentation_env(task, seed=ENV_SEED)
            env.reset()
            patched_state = patch_target_object_xy(
                env, initial_states[episode_idx], target_object, center_x, center_y
            )
            obs = env.set_init_state(patched_state)
            for _ in range(NUM_STEPS_WAIT):
                obs, _, _, _ = env.step(get_libero_dummy_action("openvla"))
            seg_keys = detect_instance_seg_keys(obs)
            verification = compute_mask_verification(
                obs, env, variant=variant, seg_keys=seg_keys, target_object=target_object
            )
            seg_plane = np.asarray(obs[seg_keys[0]])
            if seg_plane.ndim == 3:
                seg_plane = seg_plane[..., 0] if seg_plane.shape[-1] == 1 else seg_plane[..., 1]
            unique_vals = np.unique(seg_plane)
            instances = list(getattr(env.env.model, "instances_to_ids", {}).keys())
            id_map = getattr(env, "instance_to_id", {})
            target_mask, _ = build_robot_target_masks(
                obs[seg_keys[0]], obs[seg_keys[1]], env, "mask_target"
            )
            ys, xs = np.where(target_mask)
            centroid = (float(np.mean(ys)), float(np.mean(xs))) if len(xs) else None
            obj_names = list(getattr(env, "obj_of_interest", []) or [])
            per_obj = {}
            for name in obj_names:
                val = id_map.get(name)
                if val is not None:
                    m = seg_plane == val
                    per_obj[name] = {"seg_id": val, "pixels": int(m.sum())}
            verification["debug"] = {
                "obj_of_interest": obj_names,
                "per_obj_pixels": per_obj,
                "unique_seg_values": unique_vals.tolist()[:32],
                "instances_to_ids_keys": instances,
                "instance_to_id": id_map,
                "target_mask_centroid_yx": centroid,
                "segmentation_robot_id": getattr(env, "segmentation_robot_id", None),
            }
            obs_m = {k: (v.copy() if hasattr(v, "copy") else v) for k, v in obs.items()}
            apply_image_mask_to_obs_inplace(obs_m, env, variant=variant, mode=mode, seg_keys=seg_keys)
            img = np.ascontiguousarray(obs_m["agentview_image"][::-1, ::-1])
            buf = BytesIO()
            Image.fromarray(img).save(buf, format="PNG")
            # Seg overlay on flipped display rgb (diagnostic)
            overlay = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]).copy()
            seg_disp = seg_plane[::-1, ::-1]
            bowl_id = id_map.get(target_object, 1)
            overlay[seg_disp == bowl_id] = [255, 0, 0]
            obuf = BytesIO()
            Image.fromarray(overlay).save(obuf, format="PNG")
            result.update(
                {
                    "status": "ok",
                    "mask_verification": verification,
                    "frame_png_b64": base64.b64encode(buf.getvalue()).decode("ascii"),
                    "seg_overlay_png_b64": base64.b64encode(obuf.getvalue()).decode("ascii"),
                }
            )
        except Exception as exc:
            result["error"] = repr(exc)
        finally:
            if env is not None:
                env.close()
            gc.collect()
        return result

    return [run_one(job) for job in jobs]


def check_sanity_gate(
    aggregated: list[dict[str, Any]], raw_rollouts: list[dict[str, Any]] | None = None
) -> dict[str, Any]:
    """Visual + action sanity before correlation.

    Action ordering target>background often fails when background masks most of the frame;
    we also check per-pixel normalized shift and mask placement on primary target.
    """
    per_model: dict[str, Any] = {}
    all_pass = True
    raw_rollouts = raw_rollouts or []

    for model in ("openvla", "pi05", "openvla_oft", "x_vla"):
        target = next(
            (
                r
                for r in aggregated
                if r.get("model") == model
                and r.get("variant") == "mask_target"
                and r.get("mode") == "black"
                and r.get("status") == "ok"
            ),
            None,
        )
        bg = next(
            (
                r
                for r in aggregated
                if r.get("model") == model
                and r.get("variant") == "mask_background"
                and r.get("mode") == "black"
                and r.get("status") == "ok"
            ),
            None,
        )
        target_rollout = next(
            (
                r
                for r in raw_rollouts
                if r.get("model") == model and r.get("variant") == "mask_target" and r.get("status") == "ok"
            ),
            None,
        )
        if not target or not bg:
            per_model[model] = {"pass": False, "reason": "missing_condition"}
            all_pass = False
            continue

        ty = float(target["raw_l2_shift"])
        by = float(bg["raw_l2_shift"])
        mv = (target_rollout or {}).get("mask_verification") or {}
        target_pixels = max(int(mv.get("target_instance_pixels") or 0), 1)
        mask_pixels = max(int(mv.get("mask_pixels") or 0), 1)
        bg_pixels = 256 * 256 - mask_pixels
        primary_overlap = int(mv.get("primary_target_overlap") or mv.get("mask_overlaps_target") or 0)
        visual_pass = primary_overlap >= int(0.8 * target_pixels)
        norm_target = ty / (mask_pixels / (256 * 256))
        norm_bg = by / (bg_pixels / (256 * 256))
        action_pass = ty > by or norm_target > norm_bg
        passed = visual_pass and action_pass
        per_model[model] = {
            "pass": passed,
            "visual_pass": visual_pass,
            "action_pass": action_pass,
            "target_black": ty,
            "background_black": by,
            "target_gt_background": ty > by,
            "norm_target_shift": norm_target,
            "norm_background_shift": norm_bg,
            "norm_target_gt_background": norm_target > norm_bg,
            "mask_pixels": mask_pixels,
            "primary_target_overlap": primary_overlap,
            "primary_target_pixels": target_pixels,
        }
        if not passed:
            all_pass = False

    return {"pass": all_pass, "per_model": per_model}


def aggregate_rollout_results(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Mean metrics across base_idx rollouts per (model, variant, mode)."""
    from collections import defaultdict

    groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (row["model"], row["variant"], row["mode"], row.get("setting", ""))
        groups[key].append(row)

    out: list[dict[str, Any]] = []
    for (model, variant, mode, setting), items in sorted(groups.items()):
        ok = [r for r in items if r.get("status") == "ok"]
        if not ok:
            out.append(
                {
                    "model": model,
                    "variant": variant,
                    "mode": mode,
                    "setting": setting,
                    "suite": TASK_SUITE,
                    "status": "error",
                    "error": items[0].get("error", "all rollouts failed"),
                    "source": "live_rollout",
                }
            )
            continue
        raw = float(np.mean([r["raw_l2_shift"] for r in ok]))
        norm = float(np.mean([r["normalized_l2_shift"] for r in ok]))
        out.append(
            {
                "model": model,
                "variant": variant,
                "mode": mode,
                "setting": setting,
                "suite": TASK_SUITE,
                "status": "ok",
                "source": "live_rollout",
                "raw_l2_shift": raw,
                "normalized_l2_shift": norm,
                "action_sensitivity": raw,
                "n_rollouts": len(ok),
                "rollout_details": [
                    {
                        "base_idx": r["base_idx"],
                        "raw_l2_shift": r["raw_l2_shift"],
                        "normalized_l2_shift": r.get("normalized_l2_shift"),
                        "n_steps": r.get("n_steps"),
                    }
                    for r in ok
                ],
            }
        )
    return out


def build_local_proxy_scores(position_json: Path) -> list[dict[str, Any]]:
    import patchmask_behavioral as pb

    return pb.build_local_proxy_scores(position_json)


@app.local_entrypoint()
def main(
    test: bool = False,
    task_id: int = 0,
    use_proxy: bool = False,
    models: str = "openvla,pi05,openvla_oft,x_vla",
    verify_only: bool = False,
) -> None:
    from libero_task_config import get_task_preset

    preset = get_task_preset(task_id)
    model_list = [m.strip() for m in models.split(",") if m.strip()]
    if test:
        model_list = model_list[:1]
    conditions = SMOKE_CONDITIONS if test else PROBE_CONDITIONS
    base_indices = [0] if test else [0, 1, 2]
    max_probe_steps = 35 if test else 80
    frames_dir = LOCAL_ROOT / "patchmask_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"PatchMask probe task_id={task_id} test={test} verify_only={verify_only} use_proxy={use_proxy} "
        f"models={model_list} conditions={len(conditions)} bases={base_indices}"
    )

    if verify_only:
        verify_bases = [0] if test else base_indices
        jobs = []
        for variant, mode, setting_key in SMOKE_CONDITIONS:
            for base_idx in verify_bases:
                jobs.append(
                    {
                        "model": "env_only",
                        "variant": variant,
                        "mode": mode,
                        "setting_key": setting_key,
                        "task_id": task_id,
                        "base_idx": base_idx,
                        "center_x": preset.center_x,
                        "center_y": preset.center_y,
                        "target_object": preset.target_object,
                    }
                )
        rows = verify_masks_env_only.remote(jobs)
        for row in rows:
            if row.get("frame_png_b64"):
                path = frames_dir / f"verify_{row['variant']}_{row['mode']}_base{row['base_idx']}.png"
                path.write_bytes(base64.b64decode(row["frame_png_b64"]))
                print(f"Saved {path}")
            if row.get("seg_overlay_png_b64"):
                opath = frames_dir / f"verify_{row['variant']}_{row['mode']}_base{row['base_idx']}_segoverlay.png"
                opath.write_bytes(base64.b64decode(row["seg_overlay_png_b64"]))
                print(f"Saved {opath}")
            mv = row.get("mask_verification") or {}
            print(
                f"  {row['variant']}/{row['mode']} base{row['base_idx']}: "
                f"status={row['status']} mask_px={mv.get('mask_pixels')} "
                f"target_recall={mv.get('target_mask_recall')} keys={mv.get('seg_instance_keys')}"
            )
            dbg = mv.get("debug") or {}
            if dbg:
                print(f"    obj_of_interest={dbg.get('obj_of_interest')}")
                print(f"    per_obj={dbg.get('per_obj_pixels')}")
                print(f"    centroid={dbg.get('target_mask_centroid_yx')} instance_to_id={dbg.get('instance_to_id')}")
            if row.get("error"):
                print(f"    error: {row['error']}")
        return

    if use_proxy:
        pos_path = LOCAL_ROOT / f"action_sensitivity_task{task_id}.json"
        aggregated = build_local_proxy_scores(pos_path)
        raw_rows: list[dict[str, Any]] = []
    else:
        raw_rows: list[dict[str, Any]] = []
        save_frame = True

        for model in model_list:
            jobs: list[dict[str, Any]] = []
            for variant, mode, setting_key in conditions:
                for base_idx in base_indices:
                    jobs.append(
                        {
                            "variant": variant,
                            "mode": mode,
                            "setting_key": setting_key,
                            "task_id": task_id,
                            "base_idx": base_idx,
                            "center_x": preset.center_x,
                            "center_y": preset.center_y,
                            "target_object": preset.target_object,
                            "max_probe_steps": max_probe_steps,
                            "save_frame": save_frame,
                        }
                    )
                    save_frame = False

            dispatch = {
                "openvla": probe_openvla_live_batch,
                "pi05": probe_pi05_live_batch,
                "openvla_oft": probe_oft_live_batch,
                "x_vla": probe_xvla_live_batch,
            }
            probe_fn = dispatch.get(model)
            if probe_fn is None:
                raise ValueError(f"Unknown model: {model}")
            try:
                rows = probe_fn.remote(jobs)
            except Exception as exc:
                print(f"ERROR: {model} probe failed: {exc}")
                for job in jobs:
                    raw_rows.append({
                        "model": model,
                        "variant": job["variant"],
                        "mode": job["mode"],
                        "setting": job["setting_key"],
                        "suite": TASK_SUITE,
                        "task_id": job["task_id"],
                        "base_idx": job["base_idx"],
                        "status": "error",
                        "source": "live_rollout",
                        "error": repr(exc),
                    })
                continue
            raw_rows.extend(rows)

            for row in rows:
                if row.get("frame_png_b64"):
                    frame_path = frames_dir / f"{row['model']}_{row['setting']}_base{row['base_idx']}.png"
                    frame_path.write_bytes(base64.b64decode(row["frame_png_b64"]))
                    print(f"Saved masked frame {frame_path}")

        aggregated = aggregate_rollout_results(raw_rows)

    sanity = check_sanity_gate(aggregated, raw_rows if not use_proxy else None) if not use_proxy else {"pass": False, "per_model": {}}

    out = {
        "task_id": task_id,
        "suite": TASK_SUITE,
        "source": "proxy_circular" if use_proxy else "live_rollout",
        "n_conditions": len(aggregated),
        "conditions": aggregated,
        "raw_rollouts": raw_rows if not use_proxy else [],
        "sanity_gate": sanity,
    }
    fname = "patchmask_behavioral_scores.json" if use_proxy else "patchmask_behavioral_scores_live.json"
    path = LOCAL_ROOT / fname
    path.write_text(json.dumps(out, indent=2))
    ok_n = sum(1 for c in aggregated if c.get("status") == "ok")
    print(f"Wrote {path} ({ok_n}/{len(aggregated)} conditions ok)")
    if not use_proxy:
        print(f"Sanity gate (target_black > background_black): pass={sanity['pass']}")
        for model, stats in sanity.get("per_model", {}).items():
            print(f"  {model}: {stats}")
        if sanity["pass"]:
            print("Sanity gate PASSED — run: python run_live_validation.py")
        else:
            print("Sanity gate FAILED — fix masks before running run_live_validation.py")
