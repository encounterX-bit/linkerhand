#!/usr/bin/env python3
"""Safely replay action-library primitives in an explicit fixed order.

Dry-run is the default.  Hardware motion requires ``HW_ENABLE_TOKEN=1``,
``--enable-motion``, a passing ROS preflight, and an operator SPACE press in the
preview window.  ESC/Q aborts and holds the current pose; R returns to open.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

from src.comms.action_library import (
    ACTIVE_IDX,
    ActionLibrary,
    G20_OPEN_POSE,
    JOINT_COUNT,
    RESERVED_IDX,
    thumb_roundtrip_trajectory,
)
from src.comms.replay_action_group import playback_trajectory


@dataclass(frozen=True)
class SequenceSegment:
    primitive_id: int
    name: str
    frames: np.ndarray


G20_FOUR_FINGER_SPREAD_IDX = (6, 7, 8, 9)
G20_JOINT_NAMES = (
    "thumb_base", "index_base", "middle_base", "ring_base", "little_base",
    "thumb_abduction", "index_spread", "middle_spread", "ring_spread", "little_spread",
    "thumb_roll", "reserved_11", "reserved_12", "reserved_13", "reserved_14",
    "thumb_tip", "index_tip", "middle_tip", "ring_tip", "little_tip",
)


def trajectory_following_indices(
    trajectory: np.ndarray,
    *,
    spread_motion_epsilon: float = 1.0,
    include_moving_spreads: bool = True,
) -> tuple[int, ...]:
    """Return feedback channels, excluding static coupled four-finger spreads."""
    values = np.asarray(trajectory, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != JOINT_COUNT or not len(values):
        raise ValueError("trajectory must be non-empty (T,20)")
    indices = [i for i in ACTIVE_IDX if i not in G20_FOUR_FINGER_SPREAD_IDX]
    if include_moving_spreads:
        for index in G20_FOUR_FINGER_SPREAD_IDX:
            if float(np.ptp(values[:, index])) > max(
                0.0, spread_motion_epsilon
            ):
                indices.append(index)
    return tuple(sorted(indices))


def largest_following_error(
    state: np.ndarray, command: np.ndarray, indices: Sequence[int]
) -> tuple[float, int]:
    """Return the largest absolute feedback error and its command index."""
    selected = np.asarray(tuple(indices), dtype=np.int32)
    if selected.size == 0:
        raise ValueError("at least one feedback index is required")
    errors = np.abs(
        np.asarray(state, dtype=np.float32)[selected]
        - np.asarray(command, dtype=np.float32)[selected]
    )
    offset = int(np.argmax(errors))
    return float(errors[offset]), int(selected[offset])


def effective_command_lead_limit(
    default_limit: float,
    primitive_limit: Optional[float],
    hard_following_limit: float,
) -> float:
    """Resolve an action-specific soft wait line below the hard stop line."""
    requested = (
        float(default_limit)
        if primitive_limit is None
        else float(primitive_limit)
    )
    if (
        not np.isfinite(requested)
        or requested <= 0.0
        or not np.isfinite(hard_following_limit)
        or hard_following_limit <= 0.0
    ):
        raise ValueError("command-lead limits must be finite and positive")
    hard_limit = float(hard_following_limit)
    hard_headroom = min(1.0, hard_limit * 0.1)
    return min(requested, hard_limit - hard_headroom)


def parse_order(value: str) -> tuple[int, ...]:
    """Parse a non-empty comma-separated primitive ID sequence."""
    try:
        order = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("order must contain comma-separated integers") from exc
    if not order or any(item <= 0 for item in order):
        raise argparse.ArgumentTypeError("order must contain positive primitive IDs")
    return order


def primitive_id_from_key(
    key: int, primitive_ids: Sequence[int]
) -> Optional[int]:
    """Map ASCII number keys 1..9 to an available primitive ID."""
    if not ord("1") <= int(key) <= ord("9"):
        return None
    primitive_id = int(key) - ord("0")
    return primitive_id if primitive_id in primitive_ids else None


def build_sequence_segments(
    library: ActionLibrary,
    order: Sequence[int],
    *,
    start_pose: Sequence[float],
    max_step: int,
    blend_frames: int,
) -> tuple[SequenceSegment, ...]:
    """Build step-limited transitions and trajectories for offline inspection."""
    current = np.asarray(start_pose, dtype=np.float32).reshape(-1)
    if current.shape != (JOINT_COUNT,) or not np.all(np.isfinite(current)):
        raise ValueError("start_pose must contain 20 finite values")
    segments: list[SequenceSegment] = []
    for primitive_id in order:
        if primitive_id not in library.primitives:
            raise KeyError(f"unknown primitive id {primitive_id}")
        primitive = library.primitives[primitive_id]
        frames = playback_trajectory(
            primitive.trajectory,
            start_pose=current,
            max_step=max_step,
            blend_frames=blend_frames,
        )
        segments.append(SequenceSegment(primitive.id, primitive.name, frames))
        current = frames[-1]
    return tuple(segments)


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--order", type=parse_order, default=parse_order("1,2,3,4,5"))
    parser.add_argument("--side", choices=("right", "left"), default="right")
    parser.add_argument("--robot-camera", type=int, default=0)
    parser.add_argument("--rate", type=float, default=30.0)
    parser.add_argument("--max-range-step", type=int, default=10)
    parser.add_argument("--blend-frames", type=int, default=8)
    parser.add_argument("--pause-between", type=float, default=0.5)
    parser.add_argument(
        "--thumb-roundtrip-key",
        type=int,
        default=6,
        help=(
            "single-action number key for a dynamic thumb-only roundtrip; "
            "0 disables this override"
        ),
    )
    parser.add_argument(
        "--thumb-roundtrip-source-action",
        type=int,
        default=2,
        help="action whose thumb path supplies the dynamic roundtrip",
    )
    parser.add_argument(
        "--reset-before-sequence",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="return to the G20 open pose and settle before action 1",
    )
    parser.add_argument("--settle-tolerance", type=float, default=12.0)
    parser.add_argument("--settle-timeout", type=float, default=4.0)
    parser.add_argument(
        "--max-command-lead",
        type=float,
        default=18.0,
        help="pause trajectory advancement while command leads measured state by more than this",
    )
    parser.add_argument("--catchup-timeout", type=float, default=5.0)
    parser.add_argument(
        "--retry-command-period",
        type=float,
        default=0.2,
        help="re-publish the held command at this interval while waiting for feedback",
    )
    parser.add_argument("--max-following-error", type=float, default=35.0)
    parser.add_argument("--following-error-frames", type=int, default=3)
    parser.add_argument("--state-stale-seconds", type=float, default=0.5)
    parser.add_argument("--current-limit", type=int, default=100)
    parser.add_argument(
        "--thumb-current-limit",
        type=int,
        default=None,
        help="optional thumb-only torque limit; other fingers keep --current-limit",
    )
    parser.add_argument("--speed-limit", type=int, default=100)
    parser.add_argument(
        "--clear-faults-before-reset",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="clear latched finger faults after SPACE/R and before returning open",
    )
    parser.add_argument("--command-timeout", type=float, default=5.0)
    parser.add_argument("--state-timeout", type=float, default=5.0)
    parser.add_argument("--require-state", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--print-every", type=int, default=30)
    parser.add_argument("--enable-motion", action="store_true")
    return parser.parse_args(argv)


def _validate_args(args: argparse.Namespace) -> Optional[str]:
    if not 0 < args.rate <= 60:
        return "--rate must be in (0, 60]"
    if args.max_range_step <= 0 or args.blend_frames < 0:
        return "--max-range-step must be positive and --blend-frames nonnegative"
    if not 0 <= args.thumb_roundtrip_key <= 9:
        return "--thumb-roundtrip-key must be in 0..9"
    if args.pause_between < 0 or args.settle_timeout <= 0:
        return "--pause-between must be nonnegative and --settle-timeout positive"
    if (
        args.settle_tolerance <= 0
        or args.max_command_lead <= 0
        or args.max_following_error <= 0
    ):
        return "settle/command-lead/following error limits must be positive"
    if args.max_command_lead >= args.max_following_error:
        return "--max-command-lead must be lower than --max-following-error"
    if args.catchup_timeout <= 0 or args.retry_command_period <= 0:
        return "--catchup-timeout and --retry-command-period must be positive"
    if args.following_error_frames <= 0 or args.state_stale_seconds <= 0:
        return "following-error frames and state-stale seconds must be positive"
    if not 1 <= args.current_limit <= 100:
        return "--current-limit must be in configured range 1..100"
    if args.thumb_current_limit is not None and not 1 <= args.thumb_current_limit <= 100:
        return "--thumb-current-limit must be in configured range 1..100"
    if not 1 <= args.speed_limit <= 100:
        return "--speed-limit must be in configured range 1..100"
    return None


def _dry_run(library: ActionLibrary, args: argparse.Namespace) -> int:
    segments = build_sequence_segments(
        library,
        args.order,
        start_pose=G20_OPEN_POSE,
        max_step=args.max_range_step,
        blend_frames=args.blend_frames,
    )
    total = sum(len(item.frames) for item in segments)
    pauses = max(0, len(segments) - 1) * args.pause_between
    print(
        f"[library_sequence] DRY RUN: order={list(args.order)} "
        f"frames={total} nominal={total / args.rate + pauses:.2f}s; "
        f"reset_before_sequence={args.reset_before_sequence}; "
        "no ROS publisher created",
        flush=True,
    )
    every = max(1, int(args.print_every))
    for segment in segments:
        active = segment.frames[:, list(ACTIVE_IDX)]
        largest = float(np.max(np.abs(np.diff(active, axis=0)))) if len(active) > 1 else 0.0
        print(
            f"[library_sequence] {segment.primitive_id}:{segment.name} "
            f"frames={len(segment.frames)} max_step={largest:.1f} "
            f"start={np.rint(segment.frames[0]).astype(int).tolist()} "
            f"end={np.rint(segment.frames[-1]).astype(int).tolist()}",
            flush=True,
        )
        for index in range(0, len(segment.frames), every):
            print(
                f"[library_sequence] WOULD_CMD id={segment.primitive_id} "
                f"frame={index}/{len(segment.frames)}",
                flush=True,
            )
    return 0


def _camera_frame(capture: Any) -> np.ndarray:
    ok, frame = capture.read()
    if ok and frame is not None:
        return frame
    return np.zeros((480, 640, 3), dtype=np.uint8)


def _overlay(frame: np.ndarray, *, status: str) -> None:
    import cv2

    cv2.putText(frame, "G20 action-library sequence", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, status[-96:], (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 220, 255), 2)
    cv2.putText(frame, "SPACE sequence   1-9 single action   R open   Q/ESC abort", (15, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)


def _show_and_key(capture: Any, status: str, wait_ms: int = 1) -> int:
    import cv2

    frame = _camera_frame(capture)
    _overlay(frame, status=status)
    cv2.imshow("G20 action-library sequence", frame)
    return cv2.waitKey(max(1, int(wait_ms))) & 0xFF


def _run_target(
    *,
    ros: Any,
    rclpy: Any,
    capture: Any,
    args: argparse.Namespace,
    target: np.ndarray,
    label: str,
    state_clock: dict[str, float],
    best_effort_spread_feedback: bool = False,
    primitive_max_command_lead: Optional[float] = None,
) -> tuple[bool, str]:
    if ros.last_state is None:
        return False, f"{label}: no measured state"
    frames = playback_trajectory(
        target,
        start_pose=ros.last_state,
        max_step=args.max_range_step,
        blend_frames=args.blend_frames,
    )
    period = 1.0 / args.rate
    following_idx = list(
        trajectory_following_indices(
            target,
            include_moving_spreads=not best_effort_spread_feedback,
        )
    )
    ignored_spreads = [
        index for index in G20_FOUR_FINGER_SPREAD_IDX if index not in following_idx
    ]
    if ignored_spreads:
        print(
            f"[library_sequence] {label}: best-effort coupled spread feedback "
            f"q{ignored_spreads}",
            flush=True,
        )
    last_command = np.asarray(ros.last_state, dtype=np.float32)
    last_command_time = time.monotonic()
    following_bad = 0
    catchup_started: Optional[float] = None
    next_publish = time.monotonic()
    index = 0
    command_lead_limit = effective_command_lead_limit(
        args.max_command_lead,
        primitive_max_command_lead,
        args.max_following_error,
    )
    if primitive_max_command_lead is not None:
        print(
            f"[library_sequence] {label}: contact command-lead limit "
            f"{command_lead_limit:.0f} ticks; hard stop "
            f"{args.max_following_error:.0f}",
            flush=True,
        )
    while index < len(frames):
        rclpy.spin_once(ros.node, timeout_sec=0.0)
        if time.monotonic() - state_clock["updated"] > args.state_stale_seconds:
            return False, f"{label}: state stale at frame {index}"
        state = np.asarray(ros.last_state, dtype=np.float32)
        error, error_index = largest_following_error(
            state, last_command, following_idx
        )
        error_joint = f"q{error_index}/{G20_JOINT_NAMES[error_index]}"
        following_bad = following_bad + 1 if error > args.max_following_error else 0
        if following_bad >= args.following_error_frames:
            return False, f"{label}: following error {error:.0f} at {error_joint}"

        if error > command_lead_limit:
            now = time.monotonic()
            if catchup_started is None:
                catchup_started = now
                print(
                    f"[library_sequence] {label}: waiting for hand, "
                    f"command lead={error:.0f} at {error_joint}",
                    flush=True,
                )
            if now - catchup_started > args.catchup_timeout:
                return False, (
                    f"{label}: catch-up timeout, command lead={error:.0f} "
                    f"at {error_joint}"
                )
            if now - last_command_time >= args.retry_command_period:
                ros.publish_pose(
                    np.clip(np.rint(last_command), 0, 255).astype(np.int32).tolist()
                )
                last_command_time = now
            if _show_and_key(
                capture,
                f"{label} waiting lead={error:.0f} {error_joint}; ESC holds",
            ) in (ord("q"), 27):
                return False, f"{label}: stopped by operator"
            time.sleep(min(0.02, period))
            continue
        catchup_started = None

        now = time.monotonic()
        if now < next_publish:
            time.sleep(min(next_publish - now, 0.01))
            continue

        values = frames[index]
        command = np.clip(np.rint(values), 0, 255).astype(np.int32)
        command[list(RESERVED_IDX)] = 255
        ros.publish_pose(command.tolist())
        last_command = command.astype(np.float32)
        last_command_time = time.monotonic()
        if _show_and_key(capture, f"{label} {index + 1}/{len(frames)}; ESC holds") in (ord("q"), 27):
            return False, f"{label}: stopped by operator"
        index += 1
        next_publish = max(next_publish + period, time.monotonic())

    deadline = time.monotonic() + args.settle_timeout
    target_pose = np.asarray(frames[-1], dtype=np.float32)
    while time.monotonic() < deadline:
        rclpy.spin_once(ros.node, timeout_sec=0.0)
        if time.monotonic() - state_clock["updated"] > args.state_stale_seconds:
            return False, f"{label}: state stale while settling"
        state = np.asarray(ros.last_state, dtype=np.float32)
        error, error_index = largest_following_error(
            state, target_pose, following_idx
        )
        error_joint = f"q{error_index}/{G20_JOINT_NAMES[error_index]}"
        if error <= args.settle_tolerance:
            return True, f"{label}: complete, settle_error={error:.0f}"
        if _show_and_key(
            capture,
            f"{label} settling error={error:.0f} {error_joint}; ESC holds",
        ) in (ord("q"), 27):
            return False, f"{label}: stopped while settling"
        time.sleep(min(0.02, period))
    return False, (
        f"{label}: settle timeout above {args.settle_tolerance:.0f} ticks "
        f"at {error_joint}"
    )


def _prepare_open_reset(*, ros: Any, rclpy: Any, args: argparse.Namespace) -> None:
    """Clear latched faults only after an operator has requested a reset."""
    if not args.clear_faults_before_reset:
        return
    print("[library_sequence] clearing finger faults before open reset", flush=True)
    ros.publish_clear_faults()
    deadline = time.monotonic() + 0.25
    while time.monotonic() < deadline:
        rclpy.spin_once(ros.node, timeout_sec=0.02)


def _run_sequence(
    *, ros: Any, rclpy: Any, capture: Any, library: ActionLibrary,
    args: argparse.Namespace, state_clock: dict[str, float]
) -> tuple[bool, str]:
    ros.publish_session_active(True)
    try:
        if args.reset_before_sequence:
            print("[library_sequence] START reset to open pose", flush=True)
            _prepare_open_reset(ros=ros, rclpy=rclpy, args=args)
            ok, status = _run_target(
                ros=ros,
                rclpy=rclpy,
                capture=capture,
                args=args,
                target=np.stack((G20_OPEN_POSE,)),
                label="RESET OPEN",
                state_clock=state_clock,
            )
            print(f"[library_sequence] {status}", flush=True)
            if not ok:
                return False, f"pre-sequence reset failed: {status}"
        for sequence_index, primitive_id in enumerate(args.order):
            primitive = library.primitives[primitive_id]
            label = f"{sequence_index + 1}/{len(args.order)} {primitive_id}:{primitive.name}"
            print(f"[library_sequence] START {label}", flush=True)
            ok, status = _run_target(
                ros=ros,
                rclpy=rclpy,
                capture=capture,
                args=args,
                target=primitive.trajectory,
                label=label,
                state_clock=state_clock,
                best_effort_spread_feedback=(
                    primitive.best_effort_spread_feedback
                ),
                primitive_max_command_lead=primitive.max_command_lead,
            )
            print(f"[library_sequence] {status}", flush=True)
            if not ok:
                return False, status
            if sequence_index + 1 < len(args.order) and args.pause_between > 0:
                deadline = time.monotonic() + args.pause_between
                while time.monotonic() < deadline:
                    rclpy.spin_once(ros.node, timeout_sec=0.0)
                    if _show_and_key(capture, f"{label} done; next action shortly") in (ord("q"), 27):
                        return False, "stopped between actions"
                    time.sleep(0.02)
        completed = "->".join(str(item) for item in args.order)
        return True, f"sequence {completed} complete; holding final pose"
    finally:
        if rclpy.ok():
            ros.publish_session_active(False)


def _run_single_primitive(
    *, ros: Any, rclpy: Any, capture: Any, library: ActionLibrary,
    args: argparse.Namespace, state_clock: dict[str, float], primitive_id: int
) -> tuple[bool, str]:
    """Run one operator-selected primitive, then return to the key menu."""
    primitive = library.primitives[primitive_id]
    if primitive_id == args.thumb_roundtrip_key:
        source = library.primitives[args.thumb_roundtrip_source_action]
        target, nearest_frame, nearest_error = thumb_roundtrip_trajectory(
            source.trajectory, ros.last_state
        )
        label = (
            f"SELECTED {primitive_id}:thumb_action_{source.id}_roundtrip "
            f"nearest={nearest_frame + 1}/{len(source.trajectory)} "
            f"rms={nearest_error:.1f}; four fingers fixed"
        )
        best_effort_spread_feedback = False
        command_lead = source.max_command_lead
    else:
        target = primitive.trajectory
        label = f"SELECTED {primitive_id}:{primitive.name}"
        best_effort_spread_feedback = primitive.best_effort_spread_feedback
        command_lead = primitive.max_command_lead
    ros.publish_session_active(True)
    try:
        _prepare_open_reset(ros=ros, rclpy=rclpy, args=args)
        print(f"[library_sequence] START {label}", flush=True)
        return _run_target(
            ros=ros,
            rclpy=rclpy,
            capture=capture,
            args=args,
            target=target,
            label=label,
            state_clock=state_clock,
            best_effort_spread_feedback=best_effort_spread_feedback,
            primitive_max_command_lead=command_lead,
        )
    finally:
        if rclpy.ok():
            ros.publish_session_active(False)


def _hardware(library: ActionLibrary, args: argparse.Namespace) -> int:
    import cv2
    import rclpy

    from src.comms.camera_to_linkerhand import L20RosNode

    rclpy.init(args=None)
    ros = L20RosNode(args)
    capture = cv2.VideoCapture(args.robot_camera)
    state_clock = {"updated": 0.0}

    def _state_heartbeat(_message: Any) -> None:
        state_clock["updated"] = time.monotonic()

    state_watch = ros.node.create_subscription(ros.JointState, ros.state_topic, _state_heartbeat, 10)
    try:
        if not ros.wait_ready():
            return 2
        publishers = ros.node.count_publishers(ros.cmd_topic)
        if publishers > 1:
            print(
                f"[library_sequence] refusing hardware: found {publishers} publishers "
                f"on {ros.cmd_topic}; close the official GUI",
                file=sys.stderr,
            )
            return 2
        state_clock["updated"] = time.monotonic()
        ros.publish_settings(thumb_current_limit=args.thumb_current_limit)
        ros.publish_session_active(False)
        prefix = "reset open then " if args.reset_before_sequence else ""
        status = (
            f"READY; SPACE runs {prefix}order={list(args.order)}; "
            "keys 1-9 run an available action"
        )
        print(f"[library_sequence] {status}", flush=True)
        while True:
            rclpy.spin_once(ros.node, timeout_sec=0.0)
            key = _show_and_key(capture, status)
            if key in (ord("q"), 27):
                break
            if key == ord(" "):
                ok, status = _run_sequence(
                    ros=ros,
                    rclpy=rclpy,
                    capture=capture,
                    library=library,
                    args=args,
                    state_clock=state_clock,
                )
                print(f"[library_sequence] {status}", flush=True)
                if not ok and "operator" in status:
                    break
            primitive_id = primitive_id_from_key(
                key, tuple(library.primitives)
            )
            if primitive_id is not None:
                ok, status = _run_single_primitive(
                    ros=ros,
                    rclpy=rclpy,
                    capture=capture,
                    library=library,
                    args=args,
                    state_clock=state_clock,
                    primitive_id=primitive_id,
                )
                print(f"[library_sequence] {status}", flush=True)
                if not ok and "operator" in status:
                    break
            if key == ord("r"):
                ros.publish_session_active(True)
                try:
                    _prepare_open_reset(ros=ros, rclpy=rclpy, args=args)
                    ok, status = _run_target(
                        ros=ros,
                        rclpy=rclpy,
                        capture=capture,
                        args=args,
                        target=np.stack((G20_OPEN_POSE,)),
                        label="RETURN OPEN",
                        state_clock=state_clock,
                    )
                finally:
                    if rclpy.ok():
                        ros.publish_session_active(False)
                print(f"[library_sequence] {status}", flush=True)
        return 0
    except KeyboardInterrupt:
        return 130
    finally:
        if rclpy.ok():
            try:
                ros.publish_session_active(False)
            except Exception as exc:
                print(f"[library_sequence] cleanup hold warning: {exc}", file=sys.stderr)
        del state_watch
        ros.close()
        capture.release()
        cv2.destroyAllWindows()
        if rclpy.ok():
            rclpy.shutdown()


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    issue = _validate_args(args)
    if issue:
        print(f"[library_sequence] {issue}", file=sys.stderr)
        return 2
    if args.enable_motion and os.environ.get("HW_ENABLE_TOKEN") != "1":
        print(
            "[library_sequence] refusing hardware: a human must set HW_ENABLE_TOKEN=1",
            file=sys.stderr,
        )
        return 2
    try:
        library = ActionLibrary.load(args.library)
        missing = [item for item in args.order if item not in library.primitives]
        if missing:
            raise KeyError(f"missing primitive IDs {missing}")
        if (
            args.thumb_roundtrip_key
            and args.thumb_roundtrip_key not in library.primitives
        ):
            raise KeyError(
                f"thumb roundtrip key {args.thumb_roundtrip_key} is unavailable"
            )
        if (
            args.thumb_roundtrip_key
            and args.thumb_roundtrip_source_action not in library.primitives
        ):
            raise KeyError(
                "thumb roundtrip source action "
                f"{args.thumb_roundtrip_source_action} is unavailable"
            )
    except (OSError, ValueError, KeyError) as exc:
        print(f"[library_sequence] cannot load requested sequence: {exc}", file=sys.stderr)
        return 2
    print(
        "[library_sequence] order: "
        + " -> ".join(f"{item}:{library.primitives[item].name}" for item in args.order),
        flush=True,
    )
    if args.enable_motion:
        return _hardware(library, args)
    return _dry_run(library, args)


if __name__ == "__main__":
    raise SystemExit(main())
