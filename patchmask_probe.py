"""
Probe C: PatchMask occlusion action-sensitivity utilities.

Applies VLA-Trace-compatible black/background/mosaic masking to RGB images and
computes action-distribution L2 shift between masked vs unmasked inference.
"""

from __future__ import annotations

from typing import Any

import numpy as np

PATCHMASK_VARIANTS = (
    "mask_target",
    "mask_gripper",
    "mask_robot",
    "mask_robot_exc_gripper",
    "mask_background",
)
PATCHMASK_MODES = ("black", "background_fill", "mosaic")


def apply_black_mask(image: np.ndarray, mask: np.ndarray, value: int = 0) -> np.ndarray:
    out = np.asarray(image).copy()
    mask2d = np.asarray(mask, dtype=bool)
    if mask2d.ndim == 3:
        mask2d = mask2d.any(axis=-1)
    out[mask2d] = value
    return out


def apply_background_fill(image: np.ndarray, mask: np.ndarray, ring_width: int = 8) -> np.ndarray:
    out = np.asarray(image).copy()
    mask2d = np.asarray(mask, dtype=bool)
    if mask2d.ndim == 3:
        mask2d = mask2d.any(axis=-1)
    h, w = mask2d.shape
    ys, xs = np.where(mask2d)
    if len(xs) == 0:
        return out
    x0, x1 = max(0, xs.min() - ring_width), min(w - 1, xs.max() + ring_width)
    y0, y1 = max(0, ys.min() - ring_width), min(h - 1, ys.max() + ring_width)
    ring = np.zeros_like(mask2d)
    ring[y0 : y1 + 1, x0 : x1 + 1] = True
    ring &= ~mask2d
    if ring.any():
        fill = np.median(out[ring], axis=0)
        out[mask2d] = fill
    else:
        out[mask2d] = 0
    return out


def apply_mosaic(image: np.ndarray, mask: np.ndarray, block: int = 8) -> np.ndarray:
    out = np.asarray(image).copy()
    mask2d = np.asarray(mask, dtype=bool)
    if mask2d.ndim == 3:
        mask2d = mask2d.any(axis=-1)
    h, w = out.shape[:2]
    for y in range(0, h, block):
        for x in range(0, w, block):
            patch_mask = mask2d[y : y + block, x : x + block]
            if not patch_mask.any():
                continue
            patch = out[y : y + block, x : x + block]
            color = np.mean(patch.reshape(-1, patch.shape[-1]), axis=0)
            out[y : y + block, x : x + block][patch_mask] = color
    return out


def apply_patch_mask(
    image: np.ndarray,
    mask: np.ndarray,
    *,
    mode: str,
    mask_value: int = 0,
) -> np.ndarray:
    if mode == "black":
        return apply_black_mask(image, mask, mask_value)
    if mode == "background_fill":
        return apply_background_fill(image, mask)
    if mode == "mosaic":
        return apply_mosaic(image, mask)
    raise ValueError(f"unsupported mode: {mode}")


def action_stream_l2_shift(actions_a: list[np.ndarray], actions_b: list[np.ndarray]) -> float:
    """Mean per-step L2 between two action streams (equal length or truncated)."""
    if not actions_a or not actions_b:
        return float("nan")
    n = min(len(actions_a), len(actions_b))
    diffs = [float(np.linalg.norm(np.asarray(actions_a[i]) - np.asarray(actions_b[i]))) for i in range(n)]
    return float(np.mean(diffs))


def occlusion_sensitivity_from_streams(
    unmasked_actions: list[np.ndarray],
    masked_actions: list[np.ndarray],
    *,
    anchor_low: float,
    anchor_high: float,
) -> dict[str, Any]:
    raw = action_stream_l2_shift(unmasked_actions, masked_actions)
    if anchor_high <= anchor_low:
        calibrated = float("nan")
    else:
        calibrated = float(np.clip((raw - anchor_low) / (anchor_high - anchor_low), 0.0, 1.0))
    return {
        "raw_l2_shift": raw,
        "calibrated_0_1": calibrated,
        "n_steps": min(len(unmasked_actions), len(masked_actions)),
    }
