"""
LIBERO instance-segmentation PatchMask utilities (ported from VLA-Trace vla_trace/behavior/patchmask.py).

Used for live behavioral measurement — independent of vla_trace_numbers.json.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from patchmask_probe import apply_patch_mask

PATCHMASK_VARIANTS = frozenset(
    {"mask_target", "mask_gripper", "mask_robot", "mask_robot_exc_gripper", "mask_background"}
)


def detect_instance_seg_keys(obs: Mapping[str, Any]) -> tuple[str, str]:
    candidates = _integer_segmentation_candidates(obs)
    if len(candidates) < 2:
        raise RuntimeError(f"Need >=2 instance seg keys; found {candidates}")
    lower = {key: key.lower() for key in candidates}
    instance_keys = [key for key in candidates if "instance" in lower[key]]
    pool = instance_keys if len(instance_keys) >= 2 else candidates
    agent_key = next((key for key in pool if "agentview" in lower[key]), pool[0])
    wrist_key = next(
        (key for key in pool if "eye_in_hand" in lower[key] or "robot0_eye_in_hand" in lower[key]),
        None,
    )
    if wrist_key is None:
        wrist_key = next((key for key in pool if "hand" in lower[key] and "gripper" not in lower[key]), None)
    if wrist_key is None:
        wrist_key = pool[1] if pool[0] == agent_key else pool[0]
    if wrist_key == agent_key:
        wrist_key = pool[1] if pool[0] == agent_key else pool[0]
    return agent_key, wrist_key


def build_robot_target_masks(
    seg_agent: Any,
    seg_wrist: Any,
    env: Any,
    variant: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Build bool masks from LIBERO instance segmentation planes.

    Uses env.instance_to_id from SegmentationRenderEnv.reset() directly on the seg
    plane. We avoid env.get_segmentation_instances() here because its in-place robot
    collapse can remap pixels and misalign masks relative to agentview_image.
    """
    if variant not in PATCHMASK_VARIANTS:
        raise ValueError(f"Unsupported variant: {variant}")
    _ensure_segmentation_ids(env)
    plane_a = _mask_plane(np.asarray(seg_agent))
    plane_w = _mask_plane(np.asarray(seg_wrist))
    instance_to_id: dict[str, int] = getattr(env, "instance_to_id", {}) or {}
    robot_id = getattr(env, "segmentation_robot_id", None)
    robot_val = int(robot_id) + 1 if robot_id is not None else 1

    if variant == "mask_robot":
        return plane_a == robot_val, plane_w == robot_val
    if variant == "mask_gripper":
        gripper_id = getattr(env, "segmentation_gripper_id", None)
        if gripper_id is None:
            raise RuntimeError("mask_gripper requires env.segmentation_gripper_id")
        gripper_val = int(gripper_id) + 1
        return plane_a == gripper_val, plane_w == gripper_val
    if variant == "mask_robot_exc_gripper":
        gripper_id = getattr(env, "segmentation_gripper_id", None)
        if gripper_id is None:
            raise RuntimeError("mask_robot_exc_gripper requires env.segmentation_gripper_id")
        gripper_val = int(gripper_id) + 1
        return (plane_a == robot_val) & (plane_a != gripper_val), (plane_w == robot_val) & (plane_w != gripper_val)
    if variant == "mask_target":
        targets = list(getattr(env, "obj_of_interest", []) or [])
        return _mask_by_instance_names(plane_a, targets, instance_to_id), _mask_by_instance_names(
            plane_w, targets, instance_to_id
        )
    if variant == "mask_background":
        fg_a = plane_a > 0
        fg_w = plane_w > 0
        return ~fg_a, ~fg_w
    raise ValueError(variant)


def apply_image_mask_to_obs_inplace(
    obs: dict[str, Any],
    env: Any,
    *,
    variant: str,
    mode: str,
    seg_keys: tuple[str, str] | None = None,
) -> dict[str, Any]:
    if seg_keys is None:
        seg_keys = detect_instance_seg_keys(obs)
    agent_seg_key, wrist_seg_key = seg_keys
    mask_agent, mask_wrist = build_robot_target_masks(obs[agent_seg_key], obs[wrist_seg_key], env, variant)
    if "agentview_image" in obs:
        obs["agentview_image"] = _apply_mask_flipped_view(obs["agentview_image"], mask_agent, mode=mode)
    if "robot0_eye_in_hand_image" in obs:
        obs["robot0_eye_in_hand_image"] = _apply_mask_flipped_view(
            obs["robot0_eye_in_hand_image"], mask_wrist, mode=mode
        )
    return obs


def _apply_mask_flipped_view(image: Any, mask: np.ndarray, *, mode: str) -> np.ndarray:
    """Apply mask in the same 180° view OpenVLA/Pi0.5 use at inference (``[::-1, ::-1]``)."""
    img = np.ascontiguousarray(image)
    mask_f = np.ascontiguousarray(mask[::-1, ::-1])
    img_f = np.ascontiguousarray(img[::-1, ::-1])
    out_f = apply_patch_mask(img_f, mask_f, mode=mode)
    return np.ascontiguousarray(out_f[::-1, ::-1])


def compute_mask_verification(
    obs: Mapping[str, Any],
    env: Any,
    *,
    variant: str,
    seg_keys: tuple[str, str] | None = None,
    target_object: str | None = None,
) -> dict[str, Any]:
    """Debug stats: masked pixel counts and overlap with named target instance."""
    if seg_keys is None:
        seg_keys = detect_instance_seg_keys(obs)
    agent_seg_key, _ = seg_keys
    mask_agent, _ = build_robot_target_masks(obs[agent_seg_key], obs[seg_keys[1]], env, variant)
    target_mask, _ = build_robot_target_masks(obs[agent_seg_key], obs[seg_keys[1]], env, "mask_target")

    plane_a = _mask_plane(np.asarray(obs[agent_seg_key]))
    obj_names = list(getattr(env, "obj_of_interest", []) or [])
    primary = target_object or (obj_names[0] if obj_names else None)
    instance_to_id = getattr(env, "instance_to_id", {}) or {}

    target_instance_pixels = 0
    if primary and primary in instance_to_id:
        target_instance_pixels = int((plane_a == instance_to_id[primary]).sum())

    primary_overlap = 0
    if primary and primary in instance_to_id:
        primary_overlap = int((mask_agent & (plane_a == instance_to_id[primary])).sum())

    overlap = int((mask_agent & target_mask).sum())
    target_mask_pixels = int(target_mask.sum())
    mask_pixels = int(mask_agent.sum())

    return {
        "variant": variant,
        "agent_seg_key": agent_seg_key,
        "obj_of_interest": obj_names,
        "primary_target": primary,
        "instance_to_id": dict(instance_to_id),
        "mask_pixels": mask_pixels,
        "target_mask_pixels": target_mask_pixels,
        "target_instance_pixels": target_instance_pixels,
        "mask_overlaps_target": overlap,
        "primary_target_overlap": primary_overlap,
        "primary_target_recall": (
            primary_overlap / target_instance_pixels if target_instance_pixels > 0 else float("nan")
        ),
        "target_mask_recall": overlap / target_instance_pixels if target_instance_pixels > 0 else float("nan"),
        "variant_is_target": variant == "mask_target",
    }


def make_segmentation_env(task, *, seed: int = 0):
    """Create LIBERO env with instance segmentation enabled."""
    import importlib
    import os

    from libero.libero import get_libero_path

    SegmentationRenderEnv = None
    for module_name in ("libero.libero.envs.env_wrapper", "libero.libero.envs", "libero.envs"):
        try:
            mod = importlib.import_module(module_name)
            SegmentationRenderEnv = getattr(mod, "SegmentationRenderEnv", None)
            if SegmentationRenderEnv is not None:
                break
        except ImportError:
            continue
    if SegmentationRenderEnv is None:
        raise RuntimeError(
            "PatchMask requires LIBERO SegmentationRenderEnv; "
            "ensure libero.libero.envs.env_wrapper is importable."
        )

    task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env = SegmentationRenderEnv(
        bddl_file_name=task_bddl_file,
        camera_heights=256,
        camera_widths=256,
        camera_segmentations="instance",
    )
    env.seed(seed)
    return env, task.language


def _integer_segmentation_candidates(obs: Mapping[str, Any]) -> list[str]:
    keys: list[str] = []
    for key, value in obs.items():
        if not isinstance(value, np.ndarray):
            continue
        if value.ndim not in (2, 3):
            continue
        if value.ndim == 3 and value.shape[-1] not in (1, 2):
            continue
        if np.issubdtype(value.dtype, np.integer):
            keys.append(str(key))
    return sorted(keys)


def _ensure_segmentation_ids(env: Any) -> None:
    """Fill robot/gripper ids only; keep instance_to_id from env.reset()."""
    try:
        instances = list(env.env.model.instances_to_ids.keys())
    except AttributeError:
        return
    robot_keywords = ("Panda", "Robot", "robot", "UR5", "IIWA", "Sawyer", "Jaco")
    gripper_keywords = ("Gripper", "gripper")
    if getattr(env, "segmentation_robot_id", None) is None:
        for idx, name in enumerate(instances):
            if any(piece in name for piece in robot_keywords):
                env.segmentation_robot_id = idx
                break
        if getattr(env, "segmentation_robot_id", None) is None and instances:
            env.segmentation_robot_id = 0
    if getattr(env, "segmentation_gripper_id", None) is None:
        for idx, name in enumerate(instances):
            if any(piece in name for piece in gripper_keywords):
                env.segmentation_gripper_id = idx
                break
    if not getattr(env, "instance_to_id", None):
        robot_names = {"Panda0", "PandaGripper0", "RethinkMount0", "MountedPanda0"}
        env.segmentation_id_mapping = {
            idx: name
            for idx, name in enumerate(instances)
            if name not in robot_names and idx != getattr(env, "segmentation_robot_id", None)
        }
        env.instance_to_id = {name: idx + 1 for idx, name in env.segmentation_id_mapping.items()}


def _mask_plane(array: Any) -> np.ndarray:
    values = np.asarray(array)
    if values.ndim == 3 and values.shape[-1] == 1:
        return values[..., 0]
    if values.ndim == 3 and values.shape[-1] == 2:
        return values[..., 1]
    return values


def _mask_by_instance_names(plane: np.ndarray, names: list[str], instance_to_id: Mapping[str, int]) -> np.ndarray:
    mask = np.zeros(plane.shape, dtype=bool)
    for name in names:
        val = instance_to_id.get(name)
        if val is not None:
            mask |= plane == val
    return mask
