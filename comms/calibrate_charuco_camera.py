#!/usr/bin/env python3
"""Generate a printable ChArUco board or calibrate a fixed USB camera."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from .aruco_cube_pose import CameraCalibration, save_camera_calibration


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(
        "data/calibration/camera0_640x480.json"
    ))
    parser.add_argument("--generate-board", type=Path)
    parser.add_argument("--generate-only", action="store_true")
    parser.add_argument("--dictionary", default="DICT_5X5_100")
    parser.add_argument("--squares-x", type=int, default=7)
    parser.add_argument("--squares-y", type=int, default=5)
    parser.add_argument("--square-mm", type=float, default=30.0)
    parser.add_argument("--marker-mm", type=float, default=22.0)
    parser.add_argument("--dpi", type=float, default=300.0)
    parser.add_argument("--print-margin-mm", type=float, default=10.0)
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument("--camera-fps", type=int, default=30)
    parser.add_argument("--camera-fourcc", default="MJPG")
    parser.add_argument("--minimum-corners", type=int, default=8)
    parser.add_argument("--minimum-captures", type=int, default=15)
    return parser


def make_board(args: argparse.Namespace):
    dictionary_id = getattr(cv2.aruco, args.dictionary, None)
    if dictionary_id is None:
        raise ValueError(f"unknown OpenCV ArUco dictionary: {args.dictionary}")
    dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
    board = cv2.aruco.CharucoBoard(
        (args.squares_x, args.squares_y),
        float(args.square_mm),
        float(args.marker_mm),
        dictionary,
    )
    return dictionary, board


def generate_printable_board(
    path: Path, board, *, args: argparse.Namespace
) -> None:
    pixels_per_mm = float(args.dpi) / 25.4
    board_width = int(round(args.squares_x * args.square_mm * pixels_per_mm))
    board_height = int(round(args.squares_y * args.square_mm * pixels_per_mm))
    margin = int(round(args.print_margin_mm * pixels_per_mm))
    image = board.generateImage(
        (board_width, board_height), marginSize=0, borderBits=1
    )
    printable = cv2.copyMakeBorder(
        image, margin, margin, margin, margin,
        borderType=cv2.BORDER_CONSTANT, value=255,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), printable):
        raise RuntimeError(f"failed to write board image: {path}")
    print(
        f"[charuco] wrote {path}; print at 100% / actual size. "
        f"Chess square must measure {args.square_mm:.2f} mm."
    )


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


def main() -> int:
    args = build_parser().parse_args()
    _dictionary, board = make_board(args)
    if args.generate_board is not None:
        generate_printable_board(args.generate_board, board, args=args)
    if args.generate_only:
        if args.generate_board is None:
            raise SystemExit("--generate-only requires --generate-board")
        return 0

    detector = cv2.aruco.CharucoDetector(board)
    camera = cv2.VideoCapture(args.camera_index, cv2.CAP_V4L2)
    configure_camera(camera, args)
    if not camera.isOpened():
        raise RuntimeError(f"cannot open camera {args.camera_index}")

    captured_corners: list[np.ndarray] = []
    captured_ids: list[np.ndarray] = []
    image_size: tuple[int, int] | None = None
    message = "SPACE capture | C calibrate/save | Q/ESC quit"
    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                continue
            height, width = frame.shape[:2]
            image_size = (width, height)
            charuco_corners, charuco_ids, marker_corners, marker_ids = (
                detector.detectBoard(frame)
            )
            display = frame.copy()
            if marker_ids is not None:
                cv2.aruco.drawDetectedMarkers(display, marker_corners, marker_ids)
            corner_count = 0 if charuco_ids is None else len(charuco_ids)
            if charuco_ids is not None:
                cv2.aruco.drawDetectedCornersCharuco(
                    display, charuco_corners, charuco_ids
                )
            cv2.putText(
                display,
                f"captures={len(captured_corners)} corners={corner_count}",
                (18, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2,
            )
            cv2.putText(
                display, message, (18, 62),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2,
            )
            cv2.imshow("camera 0 ChArUco calibration", display)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                return 0
            if key == ord(" "):
                if charuco_ids is None or corner_count < args.minimum_corners:
                    message = (
                        f"REJECTED: need >= {args.minimum_corners} ChArUco corners"
                    )
                    continue
                captured_corners.append(np.asarray(
                    charuco_corners, dtype=np.float32
                ).copy())
                captured_ids.append(np.asarray(
                    charuco_ids, dtype=np.int32
                ).copy())
                message = (
                    f"captured {len(captured_corners)}; change board distance/tilt"
                )
            if key == ord("c"):
                if len(captured_corners) < args.minimum_captures:
                    message = (
                        f"need >= {args.minimum_captures} captures before calibration"
                    )
                    continue
                assert image_size is not None
                rms, matrix, distortion, _rvecs, _tvecs = (
                    cv2.aruco.calibrateCameraCharuco(
                        captured_corners,
                        captured_ids,
                        board,
                        image_size,
                        None,
                        None,
                    )
                )
                calibration = CameraCalibration(
                    image_width=image_size[0],
                    image_height=image_size[1],
                    camera_matrix=np.asarray(matrix, dtype=np.float64),
                    distortion_coefficients=np.asarray(
                        distortion, dtype=np.float64
                    ),
                    rms_reprojection_error=float(rms),
                )
                save_camera_calibration(args.output, calibration)
                print(
                    f"[charuco] saved {args.output}; captures="
                    f"{len(captured_corners)} rms={rms:.4f}px"
                )
                return 0
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    raise SystemExit(main())

