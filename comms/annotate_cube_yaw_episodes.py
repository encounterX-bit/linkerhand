#!/usr/bin/env python3
"""Annotate recorded G20 episodes with calibrated cube pose and relative yaw.

The recorder owns camera 0 during collection, so live pose tracking cannot open
the same device.  This offline pass uses each saved JPEG, writes a
``cube_pose.jsonl`` sidecar, and marks whether the terminal pose held the
requested yaw for enough frames.  The original samples and images are never
modified.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

from .aruco_cube_pose import (
    FacePoseEstimate,
    detect_markers,
    estimate_face_pose,
    load_camera_calibration,
    load_face_layout,
    relative_yaw_degrees,
)


@dataclass(frozen=True)
class EpisodeYawSummary:
    episode: str
    sample_count: int
    pose_valid_count: int
    pose_valid_ratio: float
    reference_frame_count: int
    final_confirm_count: int
    target_yaw_deg: float
    final_yaw_deg: float | None
    final_error_deg: float | None
    success: bool
    reason: str


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument(
        "--camera-calibration", type=Path,
        default=Path("data/calibration/camera0_640x480.json"),
    )
    parser.add_argument(
        "--layout-profile", type=Path,
        default=Path("data/calibration/aruco_cube_top_face.json"),
    )
    parser.add_argument("--target-yaw-deg", type=float, default=90.0)
    parser.add_argument("--target-tolerance-deg", type=float, default=5.0)
    parser.add_argument("--yaw-sign", type=float, choices=(-1.0, 1.0), default=1.0)
    parser.add_argument("--zero-frames", type=int, default=15)
    parser.add_argument("--confirm-frames", type=int, default=15)
    parser.add_argument("--max-reprojection-error-px", type=float, default=2.0)
    parser.add_argument("--minimum-markers", type=int, default=2)
    parser.add_argument("--minimum-valid-ratio", type=float, default=0.70)
    parser.add_argument("--yaw-ema-alpha", type=float, default=0.25)
    parser.add_argument(
        "--overwrite", action="store_true",
        help="replace existing cube_pose.jsonl and cube_pose_summary.json",
    )
    return parser


def average_rotations(rotations: Sequence[np.ndarray]) -> np.ndarray:
    """Return the nearest proper rotation to the arithmetic matrix mean."""
    if not rotations:
        raise ValueError("at least one rotation is required")
    mean = np.mean([
        np.asarray(rotation, dtype=np.float64).reshape(3, 3)
        for rotation in rotations
    ], axis=0)
    left, _singular, right = np.linalg.svd(mean)
    correction = np.eye(3)
    correction[2, 2] = np.linalg.det(left @ right)
    return left @ correction @ right


def _read_samples(path: Path) -> list[dict[str, Any]]:
    samples = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} must be a JSON object")
            samples.append(value)
    return samples


def _continuous_yaws(
    poses: Sequence[FacePoseEstimate | None],
    reference_rotation: np.ndarray,
    *,
    yaw_sign: float,
) -> list[float | None]:
    valid_indices: list[int] = []
    wrapped: list[float] = []
    for index, pose in enumerate(poses):
        if pose is None:
            continue
        valid_indices.append(index)
        wrapped.append(
            relative_yaw_degrees(reference_rotation, pose.rotation_matrix)
            * yaw_sign
        )
    unwrapped = np.degrees(np.unwrap(np.radians(wrapped)))
    result: list[float | None] = [None] * len(poses)
    for index, value in zip(valid_indices, unwrapped):
        result[index] = float(value)
    return result


def _ema(values: Sequence[float | None], alpha: float) -> list[float | None]:
    weight = float(np.clip(alpha, 0.0, 1.0))
    state: float | None = None
    result: list[float | None] = []
    for value in values:
        if value is None:
            result.append(None)
            continue
        state = float(value) if state is None else state + weight * (float(value) - state)
        result.append(state)
    return result


def summarize_episode(
    *,
    episode_name: str,
    poses: Sequence[FacePoseEstimate | None],
    filtered_yaws: Sequence[float | None],
    target_yaw_deg: float,
    target_tolerance_deg: float,
    zero_frame_count: int,
    confirm_frames: int,
    minimum_valid_ratio: float,
) -> EpisodeYawSummary:
    sample_count = len(poses)
    valid_count = sum(pose is not None for pose in poses)
    valid_ratio = valid_count / max(1, sample_count)
    final_confirm_count = 0
    for yaw in reversed(filtered_yaws):
        if (
            yaw is None
            or abs(float(yaw) - target_yaw_deg) > target_tolerance_deg
        ):
            break
        final_confirm_count += 1
        if final_confirm_count >= confirm_frames:
            break
    final_values = [
        float(value) for value in filtered_yaws[-confirm_frames:]
        if value is not None
    ]
    final_yaw = float(np.median(final_values)) if final_values else None
    final_error = (
        None if final_yaw is None else abs(final_yaw - target_yaw_deg)
    )

    reason = "success"
    success = True
    if sample_count == 0:
        success, reason = False, "empty episode"
    elif valid_count < zero_frame_count:
        success, reason = False, "not enough valid pose frames to establish zero"
    elif valid_ratio < minimum_valid_ratio:
        success, reason = False, (
            f"pose valid ratio {valid_ratio:.1%} < {minimum_valid_ratio:.1%}"
        )
    elif final_yaw is None:
        success, reason = False, "no valid terminal yaw"
    elif final_error is not None and final_error > target_tolerance_deg:
        success, reason = False, (
            f"terminal yaw error {final_error:.2f}deg > "
            f"{target_tolerance_deg:.2f}deg"
        )
    elif final_confirm_count < confirm_frames:
        success, reason = False, (
            f"terminal hold {final_confirm_count}/{confirm_frames} frames"
        )

    return EpisodeYawSummary(
        episode=episode_name,
        sample_count=sample_count,
        pose_valid_count=valid_count,
        pose_valid_ratio=valid_ratio,
        reference_frame_count=min(valid_count, zero_frame_count),
        final_confirm_count=final_confirm_count,
        target_yaw_deg=float(target_yaw_deg),
        final_yaw_deg=final_yaw,
        final_error_deg=final_error,
        success=success,
        reason=reason,
    )


def annotate_episode(
    episode_dir: Path,
    *,
    calibration,
    layout,
    args: argparse.Namespace,
) -> EpisodeYawSummary:
    samples_path = episode_dir / "samples.jsonl"
    output_path = episode_dir / "cube_pose.jsonl"
    summary_path = episode_dir / "cube_pose_summary.json"
    if not samples_path.is_file():
        raise ValueError(f"missing {samples_path}")
    if not args.overwrite and (output_path.exists() or summary_path.exists()):
        raise FileExistsError(
            f"{episode_dir} already has cube pose annotations; use --overwrite"
        )
    samples = _read_samples(samples_path)
    poses: list[FacePoseEstimate | None] = []
    reasons: list[str | None] = []
    for sample in samples:
        image_path = sample.get("image_path")
        if not image_path:
            poses.append(None)
            reasons.append("no image")
            continue
        image = cv2.imread(str(episode_dir / str(image_path)))
        if image is None:
            poses.append(None)
            reasons.append("image unreadable")
            continue
        corners, ids = detect_markers(image, layout.dictionary_name)
        pose = estimate_face_pose(
            corners, ids, layout, calibration,
            min_markers=args.minimum_markers,
        )
        if pose is None:
            poses.append(None)
            reasons.append("insufficient learned-face markers")
        elif pose.reprojection_error_px > args.max_reprojection_error_px:
            poses.append(None)
            reasons.append(
                f"reprojection {pose.reprojection_error_px:.3f}px exceeds limit"
            )
        else:
            poses.append(pose)
            reasons.append(None)

    reference_candidates = [
        pose.rotation_matrix for pose in poses if pose is not None
    ][:args.zero_frames]
    if reference_candidates:
        reference = average_rotations(reference_candidates)
        raw_yaws = _continuous_yaws(
            poses, reference, yaw_sign=args.yaw_sign
        )
        filtered_yaws = _ema(raw_yaws, args.yaw_ema_alpha)
    else:
        raw_yaws = [None] * len(poses)
        filtered_yaws = [None] * len(poses)

    summary = summarize_episode(
        episode_name=episode_dir.name,
        poses=poses,
        filtered_yaws=filtered_yaws,
        target_yaw_deg=args.target_yaw_deg,
        target_tolerance_deg=args.target_tolerance_deg,
        zero_frame_count=args.zero_frames,
        confirm_frames=args.confirm_frames,
        minimum_valid_ratio=args.minimum_valid_ratio,
    )
    with output_path.open("w", encoding="utf-8") as stream:
        for sample, pose, reason, raw_yaw, filtered_yaw in zip(
            samples, poses, reasons, raw_yaws, filtered_yaws
        ):
            value: dict[str, Any] = {
                "index": sample.get("index"),
                "timestamp": sample.get("timestamp"),
                "elapsed": sample.get("elapsed"),
                "image_path": sample.get("image_path"),
                "valid": pose is not None,
                "reason": reason,
                "relative_yaw_deg": raw_yaw,
                "filtered_yaw_deg": filtered_yaw,
                "target_yaw_deg": args.target_yaw_deg,
                "target_error_deg": (
                    None if filtered_yaw is None
                    else args.target_yaw_deg - filtered_yaw
                ),
            }
            if pose is not None:
                value.update({
                    "face_id": pose.face_id,
                    "marker_count": pose.marker_count,
                    "reprojection_error_px": pose.reprojection_error_px,
                    "tvec_mm": pose.tvec_mm.reshape(3).tolist(),
                    "rvec": pose.rvec.reshape(3).tolist(),
                    "rotation_matrix": pose.rotation_matrix.tolist(),
                })
            stream.write(json.dumps(
                value, ensure_ascii=False, separators=(",", ":")
            ) + "\n")
    summary_path.write_text(
        json.dumps(asdict(summary), indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> int:
    args = build_parser().parse_args()
    args.zero_frames = max(1, int(args.zero_frames))
    args.confirm_frames = max(1, int(args.confirm_frames))
    args.minimum_valid_ratio = float(np.clip(args.minimum_valid_ratio, 0.0, 1.0))
    session = args.session.expanduser().resolve()
    episode_dirs = sorted(
        path for path in session.glob("episode_*") if path.is_dir()
    )
    if not episode_dirs:
        raise SystemExit(f"no episode_* directories under {session}")
    calibration = load_camera_calibration(args.camera_calibration)
    layout = load_face_layout(args.layout_profile)
    summaries = []
    for episode_dir in episode_dirs:
        try:
            summary = annotate_episode(
                episode_dir,
                calibration=calibration,
                layout=layout,
                args=args,
            )
        except (ValueError, FileExistsError) as exc:
            print(f"[cube_yaw] {episode_dir.name}: SKIP: {exc}")
            continue
        summaries.append(summary)
        label = "VALID" if summary.success else "INVALID"
        yaw = (
            "none"
            if summary.final_yaw_deg is None
            else f"{summary.final_yaw_deg:+.2f}deg"
        )
        print(
            f"[cube_yaw] {summary.episode}: {label} final={yaw} "
            f"pose={summary.pose_valid_ratio:.1%} "
            f"hold={summary.final_confirm_count}/{args.confirm_frames} "
            f"reason={summary.reason}"
        )
    session_summary = {
        "schema": "g20_cube_yaw_session_summary_v1",
        "session": str(session),
        "camera_calibration": str(args.camera_calibration),
        "layout_profile": str(args.layout_profile),
        "target_yaw_deg": args.target_yaw_deg,
        "target_tolerance_deg": args.target_tolerance_deg,
        "episodes": [asdict(summary) for summary in summaries],
        "valid_episode_count": sum(summary.success for summary in summaries),
        "episode_count": len(summaries),
    }
    output = session / "cube_yaw_session_summary.json"
    output.write_text(
        json.dumps(session_summary, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"[cube_yaw] session valid="
        f"{session_summary['valid_episode_count']}/{len(summaries)}; "
        f"wrote {output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
