#!/usr/bin/env python3
"""Preview calibrated 6D pose and relative yaw of one tagged cube face.

This program never creates ROS publishers and cannot command the hand.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from .aruco_cube_pose import (
    FaceBoardLayout,
    FacePoseEstimate,
    detect_markers,
    estimate_face_pose,
    learn_face_layout,
    load_camera_calibration,
    load_face_layout,
    relative_yaw_degrees,
    save_face_layout,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--camera-calibration", type=Path, required=True,
        help="JSON written by calibrate_charuco_camera",
    )
    parser.add_argument(
        "--layout-profile", type=Path,
        default=Path("data/calibration/aruco_cube_top_face.json"),
    )
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument("--camera-fps", type=int, default=30)
    parser.add_argument("--camera-fourcc", default="MJPG")
    parser.add_argument("--dictionary", default="DICT_4X4_50")
    parser.add_argument("--face-id", type=int, choices=range(6))
    parser.add_argument("--cube-edge-mm", type=float, default=55.0)
    parser.add_argument("--marker-size-mm", type=float, default=18.0)
    parser.add_argument("--marker-center-dx-mm", type=float, default=35.0)
    parser.add_argument("--marker-center-dy-mm", type=float, default=35.0)
    parser.add_argument("--minimum-markers", type=int, default=2)
    parser.add_argument("--max-reprojection-error-px", type=float, default=2.5)
    parser.add_argument("--yaw-ema-alpha", type=float, default=0.25)
    parser.add_argument(
        "--yaw-sign", type=float, choices=(-1.0, 1.0), default=-1.0,
        help=(
            "display convention; -1 makes counter-clockwise motion in the "
            "camera image positive for the learned x-right/y-down face frame"
        ),
    )
    return parser


def configure_camera(camera: cv2.VideoCapture, args: argparse.Namespace) -> None:
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, args.camera_width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, args.camera_height)
    camera.set(cv2.CAP_PROP_FPS, args.camera_fps)
    if len(args.camera_fourcc) != 4:
        raise ValueError("--camera-fourcc must contain four characters")
    camera.set(
        cv2.CAP_PROP_FOURCC,
        cv2.VideoWriter_fourcc(*args.camera_fourcc),
    )


def _put_lines(frame: np.ndarray, lines: list[tuple[str, tuple[int, int, int]]]) -> None:
    for line_index, (value, color) in enumerate(lines):
        cv2.putText(
            frame, value, (18, 30 + 28 * line_index),
            cv2.FONT_HERSHEY_SIMPLEX, 0.63, color, 2,
        )


def _inferred_marker_size_mm(layout: FaceBoardLayout) -> float:
    edges = [
        np.linalg.norm(points[(corner_index + 1) % 4] - points[corner_index])
        for points in layout.marker_object_corners.values()
        for corner_index in range(4)
    ]
    return float(np.median(edges))


def main() -> int:
    args = build_parser().parse_args()
    calibration = load_camera_calibration(args.camera_calibration)
    layout: FaceBoardLayout | None = None
    if args.layout_profile.is_file():
        layout = load_face_layout(args.layout_profile)
        print(
            f"[cube_pose] loaded face {layout.face_id} layout from "
            f"{args.layout_profile}"
        )

    camera = cv2.VideoCapture(args.camera_index, cv2.CAP_V4L2)
    configure_camera(camera, args)
    if not camera.isOpened():
        raise RuntimeError(f"cannot open camera {args.camera_index}")

    reference_rotation: np.ndarray | None = None
    latest_pose: FacePoseEstimate | None = None
    filtered_yaw: float | None = None
    message = (
        "G learn upright top face | Z zero yaw | Q/ESC quit"
        if layout is None else
        "Z zero yaw | G relearn upright top face | Q/ESC quit"
    )
    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                continue
            if (
                frame.shape[1] != calibration.image_width
                or frame.shape[0] != calibration.image_height
            ):
                raise RuntimeError(
                    "camera resolution does not match calibration: "
                    f"live={frame.shape[1]}x{frame.shape[0]} "
                    f"calibrated={calibration.image_width}x"
                    f"{calibration.image_height}"
                )
            corners, ids = detect_markers(frame, args.dictionary)
            display = frame.copy()
            if len(ids):
                cv2.aruco.drawDetectedMarkers(
                    display,
                    [value.reshape(1, 4, 2).astype(np.float32) for value in corners],
                    ids.reshape(-1, 1),
                )

            latest_pose = None
            if layout is not None:
                latest_pose = estimate_face_pose(
                    corners, ids, layout, calibration,
                    min_markers=args.minimum_markers,
                )

            lines: list[tuple[str, tuple[int, int, int]]] = []
            if layout is None:
                lines.append(("NO FACE LAYOUT: show top face upright and press G", (0, 255, 255)))
            elif latest_pose is None:
                lines.append((f"FACE {layout.face_id}: need >= {args.minimum_markers} markers", (0, 255, 255)))
            else:
                error_ok = (
                    latest_pose.reprojection_error_px
                    <= args.max_reprojection_error_px
                )
                color = (0, 255, 0) if error_ok else (0, 0, 255)
                tvec = latest_pose.tvec_mm.reshape(3)
                lines.append((
                    f"FACE {latest_pose.face_id} markers={latest_pose.marker_count} "
                    f"reproj={latest_pose.reprojection_error_px:.2f}px",
                    color,
                ))
                lines.append((
                    f"camera xyz=({tvec[0]:+.1f}, {tvec[1]:+.1f}, {tvec[2]:+.1f}) mm",
                    color,
                ))
                cv2.drawFrameAxes(
                    display,
                    calibration.camera_matrix,
                    calibration.distortion_coefficients,
                    latest_pose.rvec,
                    latest_pose.tvec_mm,
                    20.0,
                    2,
                )
                if reference_rotation is None:
                    lines.append(("yaw not zeroed: press Z at the policy start pose", (0, 255, 255)))
                else:
                    raw_yaw = relative_yaw_degrees(
                        reference_rotation, latest_pose.rotation_matrix
                    ) * args.yaw_sign
                    if filtered_yaw is None:
                        filtered_yaw = raw_yaw
                    else:
                        alpha = float(np.clip(args.yaw_ema_alpha, 0.0, 1.0))
                        filtered_yaw += alpha * (raw_yaw - filtered_yaw)
                    lines.append((
                        f"relative yaw={raw_yaw:+.1f} deg  filtered={filtered_yaw:+.1f} deg",
                        color,
                    ))
            lines.append((message, (255, 255, 255)))
            _put_lines(display, lines)
            cv2.imshow("tagged cube calibrated 6D pose (preview only)", display)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                return 0
            if key == ord("g"):
                try:
                    layout = learn_face_layout(
                        corners,
                        ids,
                        cube_edge_mm=args.cube_edge_mm,
                        marker_size_mm=args.marker_size_mm,
                        marker_center_dx_mm=args.marker_center_dx_mm,
                        marker_center_dy_mm=args.marker_center_dy_mm,
                        dictionary_name=args.dictionary,
                        face_id=args.face_id,
                        calibration=calibration,
                    )
                except ValueError as exc:
                    message = f"LAYOUT REJECTED: {exc}"
                    continue
                save_face_layout(args.layout_profile, layout)
                reference_rotation = None
                filtered_yaw = None
                message = (
                    f"saved face {layout.face_id}; inferred black marker "
                    f"{_inferred_marker_size_mm(layout):.1f}mm; press Z"
                )
                print(
                    f"[cube_pose] saved learned face {layout.face_id} layout "
                    f"to {args.layout_profile}; inferred black-border size="
                    f"{_inferred_marker_size_mm(layout):.2f}mm"
                )
            if key == ord("z"):
                if latest_pose is None:
                    message = "ZERO REJECTED: no valid face pose"
                elif (
                    latest_pose.reprojection_error_px
                    > args.max_reprojection_error_px
                ):
                    message = (
                        "ZERO REJECTED: reprojection error "
                        f"{latest_pose.reprojection_error_px:.2f}px"
                    )
                else:
                    reference_rotation = latest_pose.rotation_matrix.copy()
                    filtered_yaw = 0.0
                    message = (
                        "yaw zeroed; camera-view counter-clockwise should approach +90"
                    )
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    raise SystemExit(main())
