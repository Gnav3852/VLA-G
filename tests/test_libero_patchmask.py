"""Unit tests for LIBERO PatchMask masking (VLA-Trace-aligned)."""

from __future__ import annotations

import numpy as np

from libero_patchmask import (
    apply_image_mask_to_obs_inplace,
    build_robot_target_masks,
    compute_mask_verification,
    detect_instance_seg_keys,
)


class MockEnv:
    def __init__(self):
        self.obj_of_interest = ["akita_black_bowl_1"]
        self.segmentation_robot_id = 0
        self.segmentation_gripper_id = 1
        self.instance_to_id = {"akita_black_bowl_1": 3, "plate_1": 4}


def test_detect_instance_seg_keys_prefers_instance_suffix():
    seg_agent = np.zeros((4, 4, 1), dtype=np.int32)
    seg_wrist = np.zeros((4, 4, 1), dtype=np.int32)
    obs = {
        "agentview_segmentation": np.zeros((4, 4, 2), dtype=np.int32),
        "robot0_eye_in_hand_segmentation": np.zeros((4, 4, 2), dtype=np.int32),
        "agentview_instance_segmentation": seg_agent,
        "robot0_eye_in_hand_instance_segmentation": seg_wrist,
    }
    assert detect_instance_seg_keys(obs) == (
        "agentview_instance_segmentation",
        "robot0_eye_in_hand_instance_segmentation",
    )


def test_mask_target_hits_bowl_not_table():
    env = MockEnv()
    seg_agent = np.zeros((8, 8, 1), dtype=np.int32)
    seg_agent[2:5, 2:5, 0] = 3  # bowl
    seg_agent[0, :] = 1  # robot strip
    seg_agent[6:, 6:] = 4  # plate (not in obj_of_interest)
    seg_wrist = np.zeros((8, 8, 1), dtype=np.int32)

    mask_agent, _ = build_robot_target_masks(seg_agent, seg_wrist, env, "mask_target")
    assert mask_agent[3, 3]
    assert not mask_agent[7, 7]
    assert not mask_agent[0, 0]


def test_mask_background_excludes_foreground():
    env = MockEnv()
    seg_agent = np.zeros((8, 8, 1), dtype=np.int32)
    seg_agent[2:5, 2:5, 0] = 3
    seg_agent[0, :] = 1
    seg_wrist = np.zeros((8, 8, 1), dtype=np.int32)

    mask_agent, _ = build_robot_target_masks(seg_agent, seg_wrist, env, "mask_background")
    assert mask_agent[6, 6]
    assert not mask_agent[3, 3]
    assert not mask_agent[0, 0]


def test_apply_mask_inplace_blackens_target():
    env = MockEnv()
    seg_agent = np.zeros((4, 4, 1), dtype=np.int32)
    seg_agent[1:3, 1:3, 0] = 3
    seg_wrist = np.zeros((4, 4, 1), dtype=np.int32)
    obs = {
        "agentview_image": np.full((4, 4, 3), 100, dtype=np.uint8),
        "robot0_eye_in_hand_image": np.full((4, 4, 3), 80, dtype=np.uint8),
        "agentview_instance_segmentation": seg_agent,
        "robot0_eye_in_hand_instance_segmentation": seg_wrist,
    }
    apply_image_mask_to_obs_inplace(obs, env, variant="mask_target", mode="black")
    # Mask is applied in flipped view; check model-facing orientation.
    assert obs["agentview_image"][::-1, ::-1][2, 2].tolist() == [0, 0, 0]
    assert obs["agentview_image"][::-1, ::-1][0, 0].tolist() == [100, 100, 100]


def test_mask_verification_recall():
    env = MockEnv()
    seg_agent = np.zeros((4, 4, 1), dtype=np.int32)
    seg_agent[1:3, 1:3, 0] = 3
    seg_wrist = np.zeros((4, 4, 1), dtype=np.int32)
    obs = {
        "agentview_instance_segmentation": seg_agent,
        "robot0_eye_in_hand_instance_segmentation": seg_wrist,
    }
    stats = compute_mask_verification(
        obs, env, variant="mask_target", target_object="akita_black_bowl_1"
    )
    assert stats["target_mask_recall"] == 1.0
    assert stats["mask_pixels"] == stats["target_instance_pixels"]
