#!/usr/bin/env python3
"""Recognize MediaPipe primitives and execute their hardcoded G20 trajectories.

The runner starts DISARMED.  In the camera window SPACE arms/disarms, Q/ESC
exits, and R clears the matcher/trajectory queue.  Hardware publishing requires
``HW_ENABLE_TOKEN=1`` and ``--enable-motion`` in addition to SPACE.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

from src.comms.action_library import (
    ActionLibrary,
    G20_OPEN_POSE,
    StreamingMatcher,
    TrajectoryExecutor,
)
from src.comms.group_action_recorder import draw_hand_overlay
from src.perception.mediapipe_source import MediaPipeHandSource
from src.perception.pipeline import HandPipeline


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--camera-index", type=int, default=2)
    parser.add_argument("--side", choices=("right", "left"), default="right")
    parser.add_argument("--rate", type=float, default=30.0)
    parser.add_argument("--min-detection-confidence", type=float, default=0.75)
    parser.add_argument("--min-tracking-confidence", type=float, default=0.75)
    parser.add_argument("--min-hand-score", type=float, default=0.5)
    parser.add_argument("--evaluation-interval", type=int, default=3)
    parser.add_argument("--confirm-evaluations", type=int, default=2)
    parser.add_argument("--match-margin", type=float, default=0.015)
    parser.add_argument("--max-range-step", type=int, default=5)
    parser.add_argument("--blend-frames", type=int, default=8)
    parser.add_argument("--queue-size", type=int, default=2)
    parser.add_argument("--hand-lost-frames", type=int, default=5)
    parser.add_argument("--current-limit", type=int, default=20)
    parser.add_argument("--speed-limit", type=int, default=35)
    parser.add_argument("--command-timeout", type=float, default=5.0)
    parser.add_argument("--state-timeout", type=float, default=5.0)
    parser.add_argument("--state-stale-seconds", type=float, default=0.5)
    parser.add_argument("--require-state", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enable-motion", action="store_true")
    parser.add_argument(
        "--auto-arm",
        action="store_true",
        help="dry-run only: begin matching immediately without pressing SPACE",
    )
    parser.add_argument("--no-open-on-exit", dest="open_on_exit", action="store_false")
    parser.set_defaults(open_on_exit=True)
    return parser.parse_args(argv)


def _overlay(frame: np.ndarray, *, armed: bool, fresh: bool, status: str, active: str) -> None:
    import cv2

    colour = (0, 220, 0) if armed else (0, 180, 255)
    cv2.putText(frame, "ARMED" if armed else "DISARMED", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, colour, 2)
    cv2.putText(frame, f"hand={'fresh' if fresh else 'lost'} active={active}", (15, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    cv2.putText(frame, status[-90:], (15, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    cv2.putText(frame, "SPACE arm/disarm  R clear  Q/ESC exit", (15, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


def _open_release(ros, executor: TrajectoryExecutor, step: int) -> None:
    if ros is None:
        return
    current = executor.last_command
    if current is None:
        current = np.asarray(ros.last_state if ros.last_state is not None else G20_OPEN_POSE, dtype=np.float32)
    target = G20_OPEN_POSE
    maximum = max(1, int(step))
    frames = int(np.ceil(float(np.max(np.abs(target - current))) / maximum)) + 1
    for _ in range(frames):
        current = current + np.clip(target - current, -maximum, maximum)
        command = np.rint(current).astype(np.int32).tolist()
        ros.publish_pose(command)
        time.sleep(0.02)
    executor.last_command = target.copy()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.rate <= 0:
        print("[library_teleop] --rate must be positive", file=sys.stderr)
        return 2
    if args.enable_motion and not os.environ.get("HW_ENABLE_TOKEN"):
        print("[library_teleop] refusing hardware: set HW_ENABLE_TOKEN manually", file=sys.stderr)
        return 2
    if args.enable_motion and args.auto_arm:
        print("[library_teleop] --auto-arm is forbidden with --enable-motion", file=sys.stderr)
        return 2
    if not 1 <= args.current_limit <= 30:
        print("[library_teleop] --current-limit must be in conservative range 1..30", file=sys.stderr)
        return 2
    if not 1 <= args.speed_limit <= 50:
        print("[library_teleop] --speed-limit must be in conservative range 1..50", file=sys.stderr)
        return 2

    try:
        library = ActionLibrary.load(args.library)
    except (OSError, ValueError, KeyError) as exc:
        print(f"[library_teleop] cannot load library: {exc}", file=sys.stderr)
        return 2
    matcher = StreamingMatcher(
        library,
        evaluation_interval=args.evaluation_interval,
        confirm_evaluations=args.confirm_evaluations,
        margin=args.match_margin,
    )
    executor = TrajectoryExecutor(
        library,
        max_step=args.max_range_step,
        blend_frames=args.blend_frames,
        queue_size=args.queue_size,
    )

    ros = None
    rclpy = None
    source = None
    armed = bool(args.auto_arm)
    status = f"loaded {len(library.primitives)} primitives"
    lost_frames = 0
    last_log = 0.0
    state_clock = {"updated": 0.0}
    state_watch = None
    try:
        if args.enable_motion:
            import rclpy as _rclpy
            from src.comms.camera_to_linkerhand import L20RosNode

            rclpy = _rclpy
            rclpy.init(args=None)
            ros = L20RosNode(args)
            if not ros.wait_ready():
                return 2
            command_publishers = ros.node.count_publishers(ros.cmd_topic)
            if command_publishers > 1:
                print(
                    f"[library_teleop] refusing hardware: found {command_publishers} "
                    f"publishers on {ros.cmd_topic}; close the official GUI",
                    file=sys.stderr,
                )
                return 2
            state_clock["updated"] = time.monotonic()

            def _state_heartbeat(_message) -> None:
                state_clock["updated"] = time.monotonic()

            state_watch = ros.node.create_subscription(
                ros.JointState, ros.state_topic, _state_heartbeat, 10
            )
            ros.publish_settings()
            ros.publish_session_active(False)
            if ros.last_state is not None:
                executor.last_command = np.asarray(ros.last_state, dtype=np.float32)
        else:
            print("[library_teleop] DRY RUN: no ROS command publisher", flush=True)

        import cv2

        source = MediaPipeHandSource(
            camera_index=args.camera_index,
            min_detection_confidence=args.min_detection_confidence,
            min_tracking_confidence=args.min_tracking_confidence,
            fps=args.rate,
        )
        pipeline = HandPipeline(source, force_side=args.side, min_score=args.min_hand_score)
        print("[library_teleop] starts DISARMED; focus window and press SPACE", flush=True)
        while True:
            if ros is not None:
                rclpy.spin_once(ros.node, timeout_sec=0.0)
            detection = source.read()
            processed = pipeline.process(detection)
            fresh = processed is not None and processed.detected and not processed.held
            if fresh:
                lost_frames = 0
                result = matcher.update(library.feature(processed.landmarks))
                if result is not None:
                    status = (
                        f"match {result.primitive_id}:{result.name} "
                        f"d={result.distance:.4f} conf={result.confidence:.2f}"
                    )
                    queued = armed and executor.enqueue(result.primitive_id)
                    print(f"[library_teleop] {status} {'QUEUED' if queued else 'PREVIEW'}", flush=True)
            else:
                lost_frames += 1
                if armed and lost_frames >= max(1, args.hand_lost_frames):
                    armed = False
                    matcher.reset()
                    executor.clear()
                    status = "hand lost; DISARMED"
                    if ros is not None:
                        ros.publish_session_active(False)
                    print(f"[library_teleop] {status}", flush=True)

            if armed:
                if (
                    ros is not None
                    and args.state_stale_seconds > 0
                    and time.monotonic() - state_clock["updated"] > args.state_stale_seconds
                ):
                    armed = False
                    matcher.reset()
                    executor.clear()
                    ros.publish_session_active(False)
                    status = "joint state stale; DISARMED"
                    print(f"[library_teleop] {status}", flush=True)
                    continue
                observed = ros.last_state if ros is not None else None
                command = executor.tick(observed)
                if command is not None:
                    if ros is not None:
                        ros.publish_pose(command)
                    now = time.monotonic()
                    if now - last_log >= 0.25:
                        last_log = now
                        print(
                            f"[library_teleop] {'PUBLISH' if ros is not None else 'WOULD_CMD'} "
                            f"primitive={executor.active_id} cmd={command}", flush=True,
                        )

            frame = draw_hand_overlay(
                source.last_frame_bgr,
                getattr(source, "last_landmarks_raw_px", None),
                fresh=fresh,
            )
            active = "none" if executor.active_id is None else str(executor.active_id)
            _overlay(frame, armed=armed, fresh=fresh, status=status, active=active)
            cv2.imshow("action library teleop", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("r"):
                matcher.reset()
                executor.clear()
                status = "matcher and queue cleared"
            if key == ord(" "):
                armed = not armed
                matcher.reset()
                executor.clear()
                status = "ARMED" if armed else "DISARMED"
                if ros is not None:
                    ros.publish_session_active(armed)
                print(f"[library_teleop] {status}", flush=True)
        return 0
    except KeyboardInterrupt:
        return 130
    finally:
        if ros is not None:
            ros.publish_session_active(False)
            if args.open_on_exit:
                print("[library_teleop] sending step-limited open release", flush=True)
                _open_release(ros, executor, args.max_range_step)
            ros.close()
        if source is not None:
            source.close()
        try:
            import cv2
            cv2.destroyAllWindows()
        except ImportError:
            pass
        if rclpy is not None and rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
