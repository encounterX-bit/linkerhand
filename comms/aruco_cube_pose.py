"""Calibrated ArUco face-board geometry and planar 6D pose estimation.

The tagged manipulation cube uses ``DICT_4X4_50`` IDs 0..23, with four
consecutive IDs on each face.  A face layout is learned once from a view where
that face is approximately upright in the image.  Learning records both the
2x2 marker placement and each printed marker's in-plane rotation, so runtime
PnP can use every detected marker corner without assuming an ID ordering.

All object-space distances are millimetres.  Camera translation estimates are
therefore millimetres as well.
"""

from __future__ import annotations

import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np


DEFAULT_DICTIONARY = "DICT_4X4_50"
VALID_CUBE_MARKER_IDS = range(24)


@dataclass(frozen=True)
class CameraCalibration:
    image_width: int
    image_height: int
    camera_matrix: np.ndarray
    distortion_coefficients: np.ndarray
    rms_reprojection_error: float | None = None


@dataclass(frozen=True)
class FaceBoardLayout:
    dictionary_name: str
    face_id: int
    cube_edge_mm: float
    marker_size_mm: float
    marker_center_dx_mm: float
    marker_center_dy_mm: float
    # Marker ID -> four object points in the exact corner order returned by
    # cv2.aruco.detectMarkers for that physical marker.
    marker_object_corners: dict[int, np.ndarray]


@dataclass(frozen=True)
class FacePoseEstimate:
    face_id: int
    marker_count: int
    rvec: np.ndarray
    tvec_mm: np.ndarray
    rotation_matrix: np.ndarray
    reprojection_error_px: float


def _validated_dictionary(dictionary_name: str):
    dictionary_id = getattr(cv2.aruco, dictionary_name, None)
    if dictionary_id is None:
        raise ValueError(f"unknown OpenCV ArUco dictionary: {dictionary_name}")
    return cv2.aruco.getPredefinedDictionary(dictionary_id)


def detect_markers(
    frame: np.ndarray, dictionary_name: str = DEFAULT_DICTIONARY
) -> tuple[list[np.ndarray], np.ndarray]:
    """Return marker corners as ``(4,2)`` arrays and flat integer IDs."""
    dictionary = _validated_dictionary(dictionary_name)
    corners, ids, _rejected = cv2.aruco.detectMarkers(frame, dictionary)
    if ids is None:
        return [], np.empty(0, dtype=np.int32)
    return (
        [np.asarray(value, dtype=np.float64).reshape(4, 2) for value in corners],
        np.asarray(ids, dtype=np.int32).reshape(-1),
    )


def dominant_face_id(ids: Sequence[int]) -> int | None:
    """Return the face with the most valid IDs, or ``None`` if none are valid."""
    counts: dict[int, int] = {}
    for marker_id in ids:
        value = int(marker_id)
        if value not in VALID_CUBE_MARKER_IDS:
            continue
        face_id = value // 4
        counts[face_id] = counts.get(face_id, 0) + 1
    if not counts:
        return None
    return max(counts, key=lambda face_id: counts[face_id])


def _quadrant_order(centers: np.ndarray) -> list[int]:
    """Return indices in top-left, top-right, bottom-left, bottom-right order."""
    if centers.shape != (4, 2):
        raise ValueError(f"expected four marker centers, got {centers.shape}")
    by_y = np.argsort(centers[:, 1])
    top = sorted(by_y[:2], key=lambda index: centers[index, 0])
    bottom = sorted(by_y[2:], key=lambda index: centers[index, 0])
    return [int(top[0]), int(top[1]), int(bottom[0]), int(bottom[1])]


def learn_face_layout(
    corners: Sequence[np.ndarray],
    ids: Sequence[int],
    *,
    cube_edge_mm: float,
    marker_size_mm: float,
    marker_center_dx_mm: float,
    marker_center_dy_mm: float,
    dictionary_name: str = DEFAULT_DICTIONARY,
    face_id: int | None = None,
    calibration: CameraCalibration | None = None,
) -> FaceBoardLayout:
    """Learn one face's ID placement and marker rotations from a camera frame.

    The selected face must show all four markers and should be approximately
    upright in the image.  The learned object frame has +x toward image-right
    and +y toward image-bottom in this reference capture.
    """
    _validated_dictionary(dictionary_name)
    if min(cube_edge_mm, marker_size_mm, marker_center_dx_mm, marker_center_dy_mm) <= 0:
        raise ValueError("cube and marker dimensions must be positive")
    if marker_size_mm >= marker_center_dx_mm or marker_size_mm >= marker_center_dy_mm:
        raise ValueError("marker size must be smaller than marker center spacing")
    if marker_center_dx_mm + marker_size_mm > cube_edge_mm + 1e-6:
        raise ValueError("horizontal marker geometry exceeds cube face")
    if marker_center_dy_mm + marker_size_mm > cube_edge_mm + 1e-6:
        raise ValueError("vertical marker geometry exceeds cube face")

    flat_ids = np.asarray(ids, dtype=np.int32).reshape(-1)
    if len(corners) != len(flat_ids):
        raise ValueError("corners and ids must have equal length")
    selected_face = dominant_face_id(flat_ids) if face_id is None else int(face_id)
    if selected_face is None or selected_face not in range(6):
        raise ValueError("could not select a valid cube face")
    expected_ids = set(range(selected_face * 4, selected_face * 4 + 4))
    selected = [
        (int(marker_id), np.asarray(marker_corners, dtype=np.float64).reshape(4, 2))
        for marker_corners, marker_id in zip(corners, flat_ids)
        if int(marker_id) in expected_ids
    ]
    if {marker_id for marker_id, _value in selected} != expected_ids:
        visible = sorted(marker_id for marker_id, _value in selected)
        raise ValueError(
            f"face {selected_face} needs all IDs {sorted(expected_ids)}; visible={visible}"
        )

    selected_corners = np.concatenate([
        marker_corners for _marker_id, marker_corners in selected
    ])
    if calibration is not None:
        selected_corners = cv2.undistortPoints(
            selected_corners.reshape(-1, 1, 2),
            calibration.camera_matrix,
            calibration.distortion_coefficients,
            P=calibration.camera_matrix,
        ).reshape(-1, 2)
    selected_corners_by_marker = list(np.split(selected_corners, 4))
    centers = np.asarray([
        value.mean(axis=0) for value in selected_corners_by_marker
    ])
    order = _quadrant_order(centers)
    center_positions = np.asarray([
        (-0.5 * marker_center_dx_mm, -0.5 * marker_center_dy_mm),
        (0.5 * marker_center_dx_mm, -0.5 * marker_center_dy_mm),
        (-0.5 * marker_center_dx_mm, 0.5 * marker_center_dy_mm),
        (0.5 * marker_center_dx_mm, 0.5 * marker_center_dy_mm),
    ], dtype=np.float64)
    # Four known marker centers determine the face homography.  Back-projecting
    # each detected corner through it learns arbitrary sticker rotation and the
    # effective black-border size instead of assuming marker edges are aligned
    # with the 2x2 center grid.
    image_to_face, _mask = cv2.findHomography(
        centers[order].astype(np.float64),
        center_positions,
        method=0,
    )
    if image_to_face is None:
        raise ValueError("cannot determine face homography from marker centers")
    object_corners: dict[int, np.ndarray] = {}
    for selected_index, (marker_id, _marker_corners) in enumerate(selected):
        face_xy = cv2.perspectiveTransform(
            selected_corners_by_marker[selected_index].reshape(-1, 1, 2),
            image_to_face,
        ).reshape(4, 2)
        object_corners[marker_id] = np.column_stack((
            face_xy,
            np.zeros(4, dtype=np.float64),
        ))

    inferred_edges = np.asarray([
        np.linalg.norm(points[(corner_index + 1) % 4] - points[corner_index])
        for points in object_corners.values()
        for corner_index in range(4)
    ])
    inferred_size = float(np.median(inferred_edges))
    if not np.isfinite(inferred_size) or inferred_size <= 0:
        raise ValueError("learned marker geometry is invalid")
    if abs(inferred_size - marker_size_mm) > max(4.0, marker_size_mm * 0.35):
        raise ValueError(
            "learned black-border size disagrees with measurement: "
            f"learned={inferred_size:.2f}mm provided={marker_size_mm:.2f}mm"
        )

    return FaceBoardLayout(
        dictionary_name=dictionary_name,
        face_id=selected_face,
        cube_edge_mm=float(cube_edge_mm),
        marker_size_mm=float(marker_size_mm),
        marker_center_dx_mm=float(marker_center_dx_mm),
        marker_center_dy_mm=float(marker_center_dy_mm),
        marker_object_corners=object_corners,
    )


def save_camera_calibration(path: Path, calibration: CameraCalibration) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "opencv_camera_calibration_v1",
        "image_width": calibration.image_width,
        "image_height": calibration.image_height,
        "camera_matrix": np.asarray(calibration.camera_matrix).tolist(),
        "distortion_coefficients": np.asarray(
            calibration.distortion_coefficients
        ).reshape(-1).tolist(),
        "rms_reprojection_error": calibration.rms_reprojection_error,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def load_camera_calibration(path: Path) -> CameraCalibration:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "opencv_camera_calibration_v1":
        raise ValueError(f"unsupported camera calibration schema in {path}")
    matrix = np.asarray(payload["camera_matrix"], dtype=np.float64)
    distortion = np.asarray(
        payload["distortion_coefficients"], dtype=np.float64
    ).reshape(-1, 1)
    if matrix.shape != (3, 3):
        raise ValueError(f"camera matrix must be 3x3, got {matrix.shape}")
    return CameraCalibration(
        image_width=int(payload["image_width"]),
        image_height=int(payload["image_height"]),
        camera_matrix=matrix,
        distortion_coefficients=distortion,
        rms_reprojection_error=(
            None
            if payload.get("rms_reprojection_error") is None
            else float(payload["rms_reprojection_error"])
        ),
    )


def save_face_layout(path: Path, layout: FaceBoardLayout) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "aruco_cube_face_layout_v1",
        "dictionary_name": layout.dictionary_name,
        "face_id": layout.face_id,
        "cube_edge_mm": layout.cube_edge_mm,
        "marker_size_mm": layout.marker_size_mm,
        "marker_center_dx_mm": layout.marker_center_dx_mm,
        "marker_center_dy_mm": layout.marker_center_dy_mm,
        "marker_object_corners": {
            str(marker_id): np.asarray(points).tolist()
            for marker_id, points in sorted(layout.marker_object_corners.items())
        },
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def load_face_layout(path: Path) -> FaceBoardLayout:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "aruco_cube_face_layout_v1":
        raise ValueError(f"unsupported cube layout schema in {path}")
    return FaceBoardLayout(
        dictionary_name=str(payload["dictionary_name"]),
        face_id=int(payload["face_id"]),
        cube_edge_mm=float(payload["cube_edge_mm"]),
        marker_size_mm=float(payload["marker_size_mm"]),
        marker_center_dx_mm=float(payload["marker_center_dx_mm"]),
        marker_center_dy_mm=float(payload["marker_center_dy_mm"]),
        marker_object_corners={
            int(marker_id): np.asarray(points, dtype=np.float64).reshape(4, 3)
            for marker_id, points in payload["marker_object_corners"].items()
        },
    )


def _pose_reprojection_error(
    object_points: np.ndarray,
    image_points: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    calibration: CameraCalibration,
) -> float:
    projected, _jacobian = cv2.projectPoints(
        object_points,
        rvec,
        tvec,
        calibration.camera_matrix,
        calibration.distortion_coefficients,
    )
    residual = projected.reshape(-1, 2) - image_points.reshape(-1, 2)
    return float(np.sqrt(np.mean(np.sum(residual * residual, axis=1))))


def estimate_face_pose(
    corners: Sequence[np.ndarray],
    ids: Sequence[int],
    layout: FaceBoardLayout,
    calibration: CameraCalibration,
    *,
    min_markers: int = 2,
) -> FacePoseEstimate | None:
    """Estimate object-to-camera pose for a learned planar face board."""
    flat_ids = np.asarray(ids, dtype=np.int32).reshape(-1)
    object_parts: list[np.ndarray] = []
    image_parts: list[np.ndarray] = []
    matched_ids: set[int] = set()
    for marker_corners, marker_id in zip(corners, flat_ids):
        value = int(marker_id)
        if value not in layout.marker_object_corners or value in matched_ids:
            continue
        object_parts.append(layout.marker_object_corners[value])
        image_parts.append(np.asarray(marker_corners, dtype=np.float64).reshape(4, 2))
        matched_ids.add(value)
    if len(matched_ids) < max(1, int(min_markers)):
        return None
    object_points = np.concatenate(object_parts).astype(np.float64)
    image_points = np.concatenate(image_parts).astype(np.float64)

    result = cv2.solvePnPGeneric(
        object_points,
        image_points,
        calibration.camera_matrix,
        calibration.distortion_coefficients,
        flags=cv2.SOLVEPNP_IPPE,
    )
    if not result or not result[0]:
        return None
    rvecs, tvecs = result[1], result[2]
    candidates: list[tuple[float, np.ndarray, np.ndarray]] = []
    for rvec, tvec in itertools.zip_longest(rvecs, tvecs):
        if rvec is None or tvec is None:
            continue
        rvec_array = np.asarray(rvec, dtype=np.float64).reshape(3, 1)
        tvec_array = np.asarray(tvec, dtype=np.float64).reshape(3, 1)
        if tvec_array[2, 0] <= 0:
            continue
        error = _pose_reprojection_error(
            object_points, image_points, rvec_array, tvec_array, calibration
        )
        candidates.append((error, rvec_array, tvec_array))
    if not candidates:
        return None
    _initial_error, rvec, tvec = min(candidates, key=lambda item: item[0])
    try:
        rvec, tvec = cv2.solvePnPRefineLM(
            object_points,
            image_points,
            calibration.camera_matrix,
            calibration.distortion_coefficients,
            rvec,
            tvec,
        )
    except cv2.error:
        pass
    error = _pose_reprojection_error(
        object_points, image_points, rvec, tvec, calibration
    )
    rotation, _jacobian = cv2.Rodrigues(rvec)
    return FacePoseEstimate(
        face_id=layout.face_id,
        marker_count=len(matched_ids),
        rvec=np.asarray(rvec, dtype=np.float64).reshape(3, 1),
        tvec_mm=np.asarray(tvec, dtype=np.float64).reshape(3, 1),
        rotation_matrix=np.asarray(rotation, dtype=np.float64),
        reprojection_error_px=error,
    )


def relative_yaw_degrees(
    reference_rotation: np.ndarray, current_rotation: np.ndarray
) -> float:
    """In-plane face rotation from a reference pose, in degrees."""
    reference = np.asarray(reference_rotation, dtype=np.float64).reshape(3, 3)
    current = np.asarray(current_rotation, dtype=np.float64).reshape(3, 3)
    relative = reference.T @ current
    return float(np.degrees(np.arctan2(relative[1, 0], relative[0, 0])))
