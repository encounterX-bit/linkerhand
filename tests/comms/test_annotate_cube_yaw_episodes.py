from __future__ import annotations

import numpy as np

from src.comms.annotate_cube_yaw_episodes import (
    _continuous_yaws,
    _ema,
    average_rotations,
    summarize_episode,
)
from src.comms.aruco_cube_pose import FacePoseEstimate


def z_rotation(degrees: float) -> np.ndarray:
    angle = np.radians(degrees)
    return np.asarray([
        [np.cos(angle), -np.sin(angle), 0.0],
        [np.sin(angle), np.cos(angle), 0.0],
        [0.0, 0.0, 1.0],
    ])


def pose(degrees: float) -> FacePoseEstimate:
    rotation = z_rotation(degrees)
    return FacePoseEstimate(
        face_id=5,
        marker_count=4,
        rvec=np.zeros((3, 1)),
        tvec_mm=np.asarray([[0.0], [0.0], [500.0]]),
        rotation_matrix=rotation,
        reprojection_error_px=0.5,
    )


def test_average_rotations_and_continuous_yaw():
    reference = average_rotations([z_rotation(-0.2), z_rotation(0.2)])
    poses = [pose(0.0), pose(45.0), None, pose(90.0)]
    yaws = _continuous_yaws(poses, reference, yaw_sign=1.0)
    assert np.allclose([yaws[0], yaws[1], yaws[3]], [0.0, 45.0, 90.0])
    assert yaws[2] is None


def test_ema_preserves_missing_frames_without_resetting_state():
    values = _ema([0.0, 10.0, None, 10.0], alpha=0.5)
    assert values == [0.0, 5.0, None, 7.5]


def test_summary_requires_terminal_hold_and_valid_ratio():
    poses = [pose(0.0)] * 15 + [pose(90.0)] * 20
    yaws = [0.0] * 15 + [90.0] * 20
    summary = summarize_episode(
        episode_name="episode_000",
        poses=poses,
        filtered_yaws=yaws,
        target_yaw_deg=90.0,
        target_tolerance_deg=5.0,
        zero_frame_count=15,
        confirm_frames=15,
        minimum_valid_ratio=0.7,
    )
    assert summary.success
    assert summary.final_yaw_deg == 90.0
    assert summary.final_confirm_count == 15

