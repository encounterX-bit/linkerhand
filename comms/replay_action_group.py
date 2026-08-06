#!/usr/bin/env python3
"""Safely preview or replay one recorded G20 action-group trajectory.

The default mode is hardware-free and prints a trajectory preview.  Real ROS2
publishing requires all of the following: ``HW_ENABLE_TOKEN=1``,
``--enable-motion``, a passing recorded command/state preflight, and a human
SPACE press in the robot-camera window.  The official GUI should be closed
during replay so there is only one command publisher.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

from src.comms.action_library import (
    ACTIVE_IDX,
    G20_OPEN_POSE,
    JOINT_COUNT,
    RESERVED_IDX,
    interpolate_waypoints,
)


@dataclass(frozen=True)
class ReplayGroup:
    path: Path
    trajectory: np.ndarray
    recorded_waypoints: int
    max_recorded_error: float
    worst_waypoint: int
    worst_joint: int
    issues: tuple[str, ...]


def _pose(value: Any, *, label: str) -> np.ndarray:
    pose = np.asarray(value, dtype=np.float32).reshape(-1)
    if pose.shape != (JOINT_COUNT,) or not np.all(np.isfinite(pose)):
        raise ValueError(f"{label} must contain 20 finite values")
    if np.any(pose < 0.0) or np.any(pose > 255.0):
        raise ValueError(f"{label} values must lie in SDK range 0..255")
    return pose


def load_replay_group(
    path: Path,
    *,
    rate: float = 30.0,
    max_recorded_state_error: float = 10.0,
) -> ReplayGroup:
    """Load one group and compute non-destructive replay preflight results."""
    group_path = Path(path).resolve()
    metadata = json.loads((group_path / "group.json").read_text(encoding="utf-8"))
    waypoint_payload = json.loads(
        (group_path / "robot" / "waypoints.json").read_text(encoding="utf-8")
    )
    trajectory_waypoints = waypoint_payload.get("trajectory_waypoints", [])
    trajectory = interpolate_waypoints(trajectory_waypoints, fps=rate)
    recorded = waypoint_payload.get("waypoints", [])

    issues: list[str] = []
    if metadata.get("status") != "complete":
        issues.append(f"group status is {metadata.get('status')!r}, not 'complete'")
    expected = int(metadata.get("robot_waypoints", 0))
    if expected != len(trajectory_waypoints):
        issues.append(
            f"group metadata says {expected} waypoints but trajectory has "
            f"{len(trajectory_waypoints)}"
        )
    if len(recorded) != len(trajectory_waypoints):
        issues.append(
            f"recorded command/state count {len(recorded)} does not match trajectory "
            f"count {len(trajectory_waypoints)}"
        )

    max_error = 0.0
    worst_waypoint = -1
    worst_joint = -1
    for index, item in enumerate(recorded):
        command = _pose(item.get("command"), label=f"waypoint {index} command")
        state = _pose(item.get("state"), label=f"waypoint {index} state")
        errors = np.abs(command - state)
        active_errors = errors[list(ACTIVE_IDX)]
        local = int(np.argmax(active_errors))
        error = float(active_errors[local])
        if error > max_error:
            max_error = error
            worst_waypoint = index
            worst_joint = int(ACTIVE_IDX[local])
    if not recorded:
        issues.append("no recorded command/state waypoints")
    elif max_error > float(max_recorded_state_error):
        issues.append(
            f"recorded command/state error {max_error:.0f} at waypoint "
            f"{worst_waypoint}, q{worst_joint} exceeds limit "
            f"{float(max_recorded_state_error):.0f}"
        )

    return ReplayGroup(
        path=group_path,
        trajectory=trajectory,
        recorded_waypoints=len(recorded),
        max_recorded_error=max_error,
        worst_waypoint=worst_waypoint,
        worst_joint=worst_joint,
        issues=tuple(issues),
    )


def densify_trajectory(
    trajectory: Sequence[Sequence[float]], *, max_step: int
) -> np.ndarray:
    """Insert frames so every active-joint command delta is at most max_step."""
    values = np.asarray(trajectory, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != JOINT_COUNT or not len(values):
        raise ValueError(f"trajectory must be non-empty (T,20), got {values.shape}")
    if max_step <= 0:
        raise ValueError("max_step must be positive")
    frames = [values[0].copy()]
    active = list(ACTIVE_IDX)
    for start, end in zip(values, values[1:]):
        largest = float(np.max(np.abs(end[active] - start[active])))
        count = max(1, int(np.ceil(largest / float(max_step))))
        for alpha in np.linspace(1.0 / count, 1.0, count):
            frames.append(((1.0 - alpha) * start + alpha * end).astype(np.float32))
    result = np.stack(frames)
    result[:, list(RESERVED_IDX)] = 255.0
    return result


def playback_trajectory(
    trajectory: np.ndarray,
    *,
    start_pose: Sequence[float],
    max_step: int,
    blend_frames: int,
) -> np.ndarray:
    """Add a safe measured-state-to-first-pose blend and step-limit all frames."""
    start = _pose(start_pose, label="start pose")
    target = np.asarray(trajectory, dtype=np.float32)
    active = list(ACTIVE_IDX)
    largest = float(np.max(np.abs(target[0, active] - start[active])))
    blend_count = max(max(0, int(blend_frames)), int(np.ceil(largest / max_step)))
    frames: list[np.ndarray] = []
    if blend_count:
        for alpha in np.linspace(1.0 / blend_count, 1.0, blend_count):
            frames.append((1.0 - alpha) * start + alpha * target[0])
    else:
        frames.append(target[0])
    if len(target) > 1:
        frames.extend(target[1:])
    return densify_trajectory(np.stack(frames), max_step=max_step)


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", type=Path, required=True)
    parser.add_argument("--side", choices=("right", "left"), default="right")
    parser.add_argument("--robot-camera", type=int, default=0)
    parser.add_argument("--rate", type=float, default=30.0)
    parser.add_argument("--max-range-step", type=int, default=5)
    parser.add_argument("--blend-frames", type=int, default=8)
    parser.add_argument("--max-recorded-state-error", type=float, default=10.0)
    parser.add_argument("--max-following-error", type=float, default=30.0)
    parser.add_argument("--following-error-frames", type=int, default=3)
    parser.add_argument("--state-stale-seconds", type=float, default=0.5)
    parser.add_argument("--current-limit", type=int, default=20)
    parser.add_argument("--speed-limit", type=int, default=35)
    parser.add_argument("--command-timeout", type=float, default=5.0)
    parser.add_argument("--state-timeout", type=float, default=5.0)
    parser.add_argument("--require-state", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--print-every", type=int, default=15)
    parser.add_argument("--enable-motion", action="store_true")
    return parser.parse_args(argv)


def _print_preflight(group: ReplayGroup, rate: float) -> None:
    duration = len(group.trajectory) / rate
    verdict = "BLOCKED" if group.issues else "REPLAYABLE"
    print(
        f"[group_replay] {verdict}: {group.path}\n"
        f"[group_replay] waypoints={group.recorded_waypoints} "
        f"interpolated_frames={len(group.trajectory)} nominal={duration:.2f}s "
        f"max_recorded_error={group.max_recorded_error:.0f} "
        f"(waypoint={group.worst_waypoint}, q{group.worst_joint})",
        flush=True,
    )
    for issue in group.issues:
        print(f"[group_replay] ISSUE: {issue}", flush=True)


def _dry_run(group: ReplayGroup, args: argparse.Namespace) -> int:
    frames = playback_trajectory(
        group.trajectory,
        start_pose=G20_OPEN_POSE,
        max_step=args.max_range_step,
        blend_frames=args.blend_frames,
    )
    every = max(1, int(args.print_every))
    print(
        f"[group_replay] DRY RUN only: playback_frames={len(frames)} "
        f"duration={len(frames) / args.rate:.2f}s; no ROS publisher created",
        flush=True,
    )
    selected = sorted(set(range(0, len(frames), every)) | {len(frames) - 1})
    for index in selected:
        command = np.rint(frames[index]).astype(np.int32).tolist()
        print(f"[group_replay] WOULD_CMD frame={index:04d} {command}", flush=True)
    return 0


def _overlay(frame: np.ndarray, *, status: str, group_name: str) -> None:
    import cv2

    cv2.putText(frame, group_name, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, status[-90:], (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 2)
    cv2.putText(
        frame,
        "SPACE replay once   R return open   Q/ESC stop",
        (15, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
    )


def _camera_frame(capture: Any) -> np.ndarray:
    ok, frame = capture.read()
    if ok and frame is not None:
        return frame
    return np.zeros((480, 640, 3), dtype=np.uint8)


def _run_motion(
    *,
    ros: Any,
    rclpy: Any,
    capture: Any,
    group: ReplayGroup,
    args: argparse.Namespace,
    target: np.ndarray,
    label: str,
    state_clock: dict[str, float],
) -> tuple[bool, str]:
    import cv2

    if ros.last_state is None:
        return False, "no measured state; replay aborted"
    frames = playback_trajectory(
        target,
        start_pose=ros.last_state,
        max_step=args.max_range_step,
        blend_frames=args.blend_frames,
    )
    period = 1.0 / args.rate
    following_bad = 0
    last_command = np.asarray(ros.last_state, dtype=np.float32)
    ros.publish_session_active(True)
    try:
        for index, values in enumerate(frames):
            started = time.monotonic()
            rclpy.spin_once(ros.node, timeout_sec=0.0)
            if time.monotonic() - state_clock["updated"] > args.state_stale_seconds:
                return False, f"state stale at frame {index}; stopped"
            state = np.asarray(ros.last_state, dtype=np.float32)
            following = float(
                np.max(np.abs(state[list(ACTIVE_IDX)] - last_command[list(ACTIVE_IDX)]))
            )
            following_bad = following_bad + 1 if following > args.max_following_error else 0
            if following_bad >= args.following_error_frames:
                return False, f"following error {following:.0f} at frame {index}; stopped"

            command = np.rint(values).astype(np.int32)
            command[list(RESERVED_IDX)] = 255
            ros.publish_pose(command.tolist())
            last_command = command.astype(np.float32)

            frame = _camera_frame(capture)
            _overlay(
                frame,
                status=f"{label} RUNNING {index + 1}/{len(frames)}; ESC stops",
                group_name=group.path.name,
            )
            cv2.imshow("G20 action-group replay", frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                return False, f"{label} stopped by operator"
            remaining = period - (time.monotonic() - started)
            if remaining > 0:
                time.sleep(remaining)
        return True, f"{label} complete; DISARMED"
    finally:
        ros.publish_session_active(False)


def _hardware(group: ReplayGroup, args: argparse.Namespace) -> int:
    import cv2
    import rclpy

    from src.comms.camera_to_linkerhand import L20RosNode

    rclpy.init(args=None)
    ros = L20RosNode(args)
    capture = cv2.VideoCapture(args.robot_camera)
    state_clock = {"updated": 0.0}

    def _state_heartbeat(_message: Any) -> None:
        state_clock["updated"] = time.monotonic()

    state_watch = ros.node.create_subscription(
        ros.JointState, ros.state_topic, _state_heartbeat, 10
    )
    try:
        if not ros.wait_ready():
            return 2
        command_publishers = ros.node.count_publishers(ros.cmd_topic)
        if command_publishers > 1:
            print(
                f"[group_replay] refusing hardware: found {command_publishers} "
                f"publishers on {ros.cmd_topic}; close the official GUI",
                file=sys.stderr,
            )
            return 2
        state_clock["updated"] = time.monotonic()
        ros.publish_settings()
        ros.publish_session_active(False)
        status = "DISARMED; focus this window and press SPACE"
        print(f"[group_replay] {status}", flush=True)
        while True:
            rclpy.spin_once(ros.node, timeout_sec=0.0)
            frame = _camera_frame(capture)
            _overlay(frame, status=status, group_name=group.path.name)
            cv2.imshow("G20 action-group replay", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord(" "):
                ok, status = _run_motion(
                    ros=ros,
                    rclpy=rclpy,
                    capture=capture,
                    group=group,
                    args=args,
                    target=group.trajectory,
                    label="REPLAY",
                    state_clock=state_clock,
                )
                print(f"[group_replay] {status}", flush=True)
                if not ok:
                    print("[group_replay] motion stopped; inspect the hand before retrying", flush=True)
                    if "stopped by operator" in status:
                        break
            if key == ord("r"):
                ok, status = _run_motion(
                    ros=ros,
                    rclpy=rclpy,
                    capture=capture,
                    group=group,
                    args=args,
                    target=np.stack((G20_OPEN_POSE,)),
                    label="RETURN OPEN",
                    state_clock=state_clock,
                )
                print(f"[group_replay] {status}", flush=True)
                if not ok:
                    print("[group_replay] open return stopped; inspect the hand", flush=True)
        return 0
    except KeyboardInterrupt:
        return 130
    finally:
        ros.publish_session_active(False)
        del state_watch
        ros.close()
        capture.release()
        cv2.destroyAllWindows()
        if rclpy.ok():
            rclpy.shutdown()


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    if args.rate <= 0 or args.rate > 60:
        print("[group_replay] --rate must be in (0, 60]", file=sys.stderr)
        return 2
    if args.max_range_step <= 0:
        print("[group_replay] --max-range-step must be positive", file=sys.stderr)
        return 2
    if args.following_error_frames <= 0:
        print("[group_replay] --following-error-frames must be positive", file=sys.stderr)
        return 2
    if args.state_stale_seconds <= 0 or args.max_following_error <= 0:
        print(
            "[group_replay] state-stale and following-error limits must be positive",
            file=sys.stderr,
        )
        return 2
    if not 1 <= args.current_limit <= 30:
        print("[group_replay] --current-limit must be in conservative range 1..30", file=sys.stderr)
        return 2
    if not 1 <= args.speed_limit <= 50:
        print("[group_replay] --speed-limit must be in conservative range 1..50", file=sys.stderr)
        return 2
    if args.enable_motion and os.environ.get("HW_ENABLE_TOKEN") != "1":
        print(
            "[group_replay] refusing hardware: a human must set HW_ENABLE_TOKEN=1",
            file=sys.stderr,
        )
        return 2

    try:
        group = load_replay_group(
            args.group,
            rate=args.rate,
            max_recorded_state_error=args.max_recorded_state_error,
        )
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"[group_replay] cannot load group: {exc}", file=sys.stderr)
        return 2
    _print_preflight(group, args.rate)
    if args.enable_motion:
        if group.issues:
            print("[group_replay] refusing hardware because preflight is BLOCKED", file=sys.stderr)
            return 2
        return _hardware(group, args)
    return _dry_run(group, args)


if __name__ == "__main__":
    raise SystemExit(main())
