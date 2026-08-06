from __future__ import annotations

import cv2
import numpy as np

from src.comms.aruco_cube_pose import (
    CameraCalibration,
    estimate_face_pose,
    learn_face_layout,
    relative_yaw_degrees,
)


def synthetic_face_pixels():
    centers = [
        np.asarray([200.0, 140.0]),
        np.asarray([300.0, 140.0]),
        np.asarray([200.0, 240.0]),
        np.asarray([300.0, 240.0]),
    ]
    # 100 px center spacing represents 35 mm, so an 18 mm marker is
    # 51.428... px wide. Rotate every printed marker 23 degrees relative to the
    # center grid to exercise continuous (not merely 90-degree) sticker angles.
    half_pixels = 0.5 * 18.0 / 35.0 * 100.0
    canonical = np.asarray([
        [-half_pixels, -half_pixels],
        [half_pixels, -half_pixels],
        [half_pixels, half_pixels],
        [-half_pixels, half_pixels],
    ])
    angle = np.radians(23.0)
    in_plane_rotation = np.asarray([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)],
    ])
    canonical = canonical @ in_plane_rotation.T
    shifts = [0, 1, 2, 3]
    ids = np.asarray([8, 9, 10, 11], dtype=np.int32)
    corners = [
        center + np.roll(canonical, -shift, axis=0)
        for center, shift in zip(centers, shifts)
    ]
    return corners, ids


def test_learn_face_layout_recovers_grid_and_marker_rotations():
    corners, ids = synthetic_face_pixels()
    layout = learn_face_layout(
        corners,
        ids,
        cube_edge_mm=55,
        marker_size_mm=18,
        marker_center_dx_mm=35,
        marker_center_dy_mm=35,
    )
    assert layout.face_id == 2
    assert set(layout.marker_object_corners) == {8, 9, 10, 11}
    centers = np.asarray([
        points.mean(axis=0)[:2]
        for points in layout.marker_object_corners.values()
    ])
    assert set(map(tuple, np.round(centers, 5))) == {
        (-17.5, -17.5),
        (17.5, -17.5),
        (-17.5, 17.5),
        (17.5, 17.5),
    }
    edge_lengths = [
        np.linalg.norm(points[(index + 1) % 4] - points[index])
        for points in layout.marker_object_corners.values()
        for index in range(4)
    ]
    assert np.allclose(edge_lengths, 18.0)


def test_estimate_face_pose_recovers_synthetic_pnp():
    reference_corners, ids = synthetic_face_pixels()
    layout = learn_face_layout(
        reference_corners,
        ids,
        cube_edge_mm=55,
        marker_size_mm=18,
        marker_center_dx_mm=35,
        marker_center_dy_mm=35,
    )
    calibration = CameraCalibration(
        image_width=640,
        image_height=480,
        camera_matrix=np.asarray([
            [620.0, 0.0, 320.0],
            [0.0, 615.0, 240.0],
            [0.0, 0.0, 1.0],
        ]),
        distortion_coefficients=np.zeros((5, 1)),
    )
    expected_rvec = np.asarray([[0.18], [-0.12], [0.34]])
    expected_tvec = np.asarray([[15.0], [-8.0], [420.0]])
    projected = []
    for marker_id in ids:
        points, _jacobian = cv2.projectPoints(
            layout.marker_object_corners[int(marker_id)],
            expected_rvec,
            expected_tvec,
            calibration.camera_matrix,
            calibration.distortion_coefficients,
        )
        projected.append(points.reshape(4, 2))

    pose = estimate_face_pose(projected, ids, layout, calibration)
    assert pose is not None
    assert pose.marker_count == 4
    assert pose.reprojection_error_px < 1e-4
    assert np.allclose(pose.tvec_mm, expected_tvec, atol=1e-3)
    expected_rotation, _jacobian = cv2.Rodrigues(expected_rvec)
    assert np.allclose(pose.rotation_matrix, expected_rotation, atol=1e-4)


def test_relative_yaw_has_expected_sign():
    angle = np.radians(90.0)
    rotation = np.asarray([
        [np.cos(angle), -np.sin(angle), 0.0],
        [np.sin(angle), np.cos(angle), 0.0],
        [0.0, 0.0, 1.0],
    ])
    assert np.isclose(relative_yaw_degrees(np.eye(3), rotation), 90.0)
