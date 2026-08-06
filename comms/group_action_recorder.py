#!/usr/bin/env python3
"""Record grouped MediaPipe repetitions and official-GUI G20 waypoints.

Keyboard workflow in the preview window::

    IDLE --SPACE--> HUMAN_READY
    HUMAN_READY --M--> HUMAN_RECORDING --M--> HUMAN_READY (repeat many takes)
    HUMAN_READY --SPACE--> ROBOT_CAPTURE
    ROBOT --S...--> capture command + state + robot-camera photo
    ROBOT --SPACE--> finalize group and immediately start the next HUMAN group

M toggles each repeated human take on/off. Q re-records human data, E re-records
robot waypoints, and X/ESC exits.
This process creates ROS subscriptions only; it never creates a command publisher.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import numpy as np

from src.perception.mediapipe_source import MediaPipeHandSource
from src.perception.pipeline import HandPipeline


JOINT_COUNT = 20
RESERVED_IDX = (11, 12, 13, 14)
HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
)


class Phase(str, Enum):
    IDLE = "idle"
    HUMAN_READY = "human_ready"
    HUMAN_RECORDING = "human_recording"
    ROBOT = "robot_capture"


@dataclass(frozen=True)
class WorkflowEvent:
    action: str
    group_index: Optional[int] = None


class GroupWorkflow:
    """Pure keyboard state machine, separated for headless testing."""

    def __init__(self, start_index: int = 0) -> None:
        self.phase = Phase.IDLE
        self.group_index = int(start_index)
        self.human_takes = 0
        self.robot_waypoints = 0

    def space(self) -> WorkflowEvent:
        if self.phase == Phase.IDLE:
            self.phase = Phase.HUMAN_READY
            self.human_takes = 0
            self.robot_waypoints = 0
            return WorkflowEvent("start_group", self.group_index)
        if self.phase == Phase.HUMAN_RECORDING:
            return WorkflowEvent("need_stop_take", self.group_index)
        if self.phase == Phase.HUMAN_READY:
            if self.human_takes == 0:
                return WorkflowEvent("need_human_take", self.group_index)
            self.phase = Phase.ROBOT
            return WorkflowEvent("stop_human", self.group_index)
        if self.robot_waypoints == 0:
            return WorkflowEvent("need_waypoint", self.group_index)
        completed = self.group_index
        self.group_index += 1
        self.phase = Phase.HUMAN_READY
        self.human_takes = 0
        self.robot_waypoints = 0
        return WorkflowEvent("finalize_and_start", completed)

    def toggle_human_take(self) -> WorkflowEvent:
        if self.phase == Phase.HUMAN_READY:
            self.phase = Phase.HUMAN_RECORDING
            return WorkflowEvent("start_take", self.group_index)
        if self.phase == Phase.HUMAN_RECORDING:
            self.phase = Phase.HUMAN_READY
            self.human_takes += 1
            return WorkflowEvent("stop_take", self.group_index)
        return WorkflowEvent("take_ignored", self.group_index)

    def waypoint(self) -> bool:
        if self.phase != Phase.ROBOT:
            return False
        self.robot_waypoints += 1
        return True

    def redo_human(self) -> None:
        self.phase = Phase.HUMAN_READY
        self.human_takes = 0

    def redo_robot(self) -> None:
        self.phase = Phase.ROBOT
        self.robot_waypoints = 0


class GuiCommandMonitor:
    """Read-only ROS monitor for the official LinkerHand GUI and driver."""

    def __init__(self, side: str) -> None:
        import rclpy
        from rclpy.node import Node
        from sensor_msgs.msg import JointState

        self.rclpy = rclpy
        self.node = Node("g20_group_action_recorder")
        self.command_topic = f"/cb_{side}_hand_control_cmd"
        self.state_topic = f"/cb_{side}_hand_state"
        self.command: Optional[list[float]] = None
        self.state: Optional[list[float]] = None
        self.command_at = 0.0
        self.state_at = 0.0
        self._command_sub = self.node.create_subscription(
            JointState, self.command_topic, self._command_cb, 10
        )
        self._state_sub = self.node.create_subscription(
            JointState, self.state_topic, self._state_cb, 10
        )

    @staticmethod
    def _position(message) -> Optional[list[float]]:
        values = np.asarray(message.position, dtype=float).reshape(-1)
        if len(values) < JOINT_COUNT or not np.all(np.isfinite(values[:JOINT_COUNT])):
            return None
        return np.clip(values[:JOINT_COUNT], 0.0, 255.0).tolist()

    def _command_cb(self, message) -> None:
        values = self._position(message)
        if values is not None:
            for index in RESERVED_IDX:
                values[index] = 255.0
            self.command = values
            self.command_at = time.monotonic()

    def _state_cb(self, message) -> None:
        values = self._position(message)
        if values is not None:
            self.state = values
            self.state_at = time.monotonic()

    def spin(self) -> None:
        self.rclpy.spin_once(self.node, timeout_sec=0.0)

    def snapshot(self, stale_seconds: float, require_state: bool) -> dict[str, Any]:
        now = time.monotonic()
        command_age = None if self.command is None else now - self.command_at
        state_age = None if self.state is None else now - self.state_at
        if command_age is None or command_age > stale_seconds:
            raise RuntimeError(f"no fresh GUI command on {self.command_topic}")
        if require_state and (state_age is None or state_age > stale_seconds):
            raise RuntimeError(f"no fresh hardware state on {self.state_topic}")
        return {
            "command": list(self.command),
            "state": None if self.state is None else list(self.state),
            "command_age_seconds": command_age,
            "state_age_seconds": state_age,
        }

    def close(self) -> None:
        self.node.destroy_node()


class GroupCapture:
    def __init__(self, session_dir: Path, index: int, jpeg_quality: int) -> None:
        self.index = int(index)
        self.path = session_dir / f"group_{index:03d}"
        self.human_image_dir = self.path / "human" / "images"
        self.robot_image_dir = self.path / "robot" / "images"
        self.human_image_dir.mkdir(parents=True, exist_ok=False)
        self.robot_image_dir.mkdir(parents=True, exist_ok=False)
        self.samples_path = self.path / "human" / "samples.jsonl"
        self.samples_file = self.samples_path.open("w", encoding="utf-8", buffering=1)
        self.jpeg_quality = int(jpeg_quality)
        self.started_wall = time.time()
        self.started_mono = time.monotonic()
        self.human_started_mono = self.started_mono
        self.human_stopped_mono: Optional[float] = None
        self.sample_count = 0
        self.fresh_count = 0
        self.markers: list[dict[str, Any]] = []
        self.human_takes: list[dict[str, Any]] = []
        self.active_take: Optional[dict[str, Any]] = None
        self.waypoints: list[dict[str, Any]] = []

    def add_human(self, frame: np.ndarray, processed, source) -> bool:
        import cv2

        image_name = f"{self.sample_count:06d}.jpg"
        image_path = self.human_image_dir / image_name
        if not cv2.imwrite(
            str(image_path), frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality],
        ):
            return False
        fresh = bool(processed is not None and processed.detected and not processed.held)
        if fresh:
            self.fresh_count += 1
        sample = {
            "index": self.sample_count,
            "timestamp": time.time(),
            "elapsed": time.monotonic() - self.human_started_mono,
            "image_path": f"images/{image_name}",
            "detected": bool(processed is not None and processed.detected),
            "held": bool(processed is not None and processed.held),
            "fresh": fresh,
            "score": None if processed is None else float(processed.score),
            "side": None if processed is None else processed.side,
            "landmarks_hand_base": (
                None
                if processed is None
                else np.asarray(processed.landmarks, dtype=float).tolist()
            ),
            "landmarks_world_raw": _optional_points(
                getattr(source, "last_world_landmarks_raw", None)
            ),
            "landmarks_px": _optional_points(
                getattr(source, "last_landmarks_raw_px", None)
            ),
        }
        self.samples_file.write(json.dumps(sample, ensure_ascii=False, separators=(",", ":")) + "\n")
        self.sample_count += 1
        return True

    def start_human_take(self) -> int:
        if self.active_take is not None:
            raise RuntimeError("a human take is already recording")
        take_index = len(self.human_takes)
        self.active_take = {
            "take_index": take_index,
            "start_sample": self.sample_count,
            "start_elapsed": time.monotonic() - self.human_started_mono,
        }
        return take_index

    def finish_human_take(self) -> dict[str, Any]:
        if self.active_take is None:
            raise RuntimeError("no human take is recording")
        item = dict(self.active_take)
        item.update({
            "end_sample": self.sample_count,
            "end_elapsed": time.monotonic() - self.human_started_mono,
            "frames": self.sample_count - int(item["start_sample"]),
        })
        self.human_takes.append(item)
        self.markers.append({
            "sample_index": self.sample_count,
            "elapsed": item["end_elapsed"],
            "take_index": item["take_index"],
        })
        self.active_take = None
        self.human_stopped_mono = time.monotonic()
        return item

    def stop_human(self) -> None:
        if self.active_take is not None:
            self.finish_human_take()
        if not self.samples_file.closed:
            self.samples_file.flush()
        self.human_stopped_mono = time.monotonic()

    def _revision_path(self, prefix: str) -> Path:
        revisions = self.path / "revisions"
        revisions.mkdir(exist_ok=True)
        index = 0
        while (revisions / f"{prefix}_retry_{index:03d}").exists():
            index += 1
        return revisions / f"{prefix}_retry_{index:03d}"

    def reset_human(self) -> Path:
        """Archive the current human attempt and start an empty replacement."""
        if not self.samples_file.closed:
            self.samples_file.close()
        human_dir = self.path / "human"
        revision = self._revision_path("human")
        shutil.move(str(human_dir), str(revision))
        self.human_image_dir = human_dir / "images"
        self.human_image_dir.mkdir(parents=True, exist_ok=False)
        self.samples_path = human_dir / "samples.jsonl"
        self.samples_file = self.samples_path.open("w", encoding="utf-8", buffering=1)
        self.sample_count = 0
        self.fresh_count = 0
        self.markers = []
        self.human_takes = []
        self.active_take = None
        self.human_started_mono = time.monotonic()
        self.human_stopped_mono = None
        return revision

    def reset_robot(self) -> Path:
        """Archive all current robot snapshots and start an empty replacement."""
        robot_dir = self.path / "robot"
        revision = self._revision_path("robot")
        shutil.move(str(robot_dir), str(revision))
        self.robot_image_dir = robot_dir / "images"
        self.robot_image_dir.mkdir(parents=True, exist_ok=False)
        self.waypoints = []
        return revision

    def add_robot_waypoint(
        self,
        frame: np.ndarray,
        snapshot: dict[str, Any],
        *,
        suggested_duration: float,
    ) -> Path:
        import cv2

        index = len(self.waypoints)
        image_name = f"waypoint_{index:03d}.jpg"
        path = self.robot_image_dir / image_name
        if not cv2.imwrite(
            str(path), frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality],
        ):
            raise RuntimeError(f"failed to write robot image {path}")
        item = {
            "index": index,
            "timestamp": time.time(),
            "elapsed_since_group_start": time.monotonic() - self.started_mono,
            "image_path": f"images/{image_name}",
            "command": snapshot["command"],
            "state": snapshot["state"],
            "command_age_seconds": snapshot["command_age_seconds"],
            "state_age_seconds": snapshot["state_age_seconds"],
            "suggested_duration": None if index == 0 else float(suggested_duration),
        }
        self.waypoints.append(item)
        self._write_waypoints()
        return path

    def _write_waypoints(self) -> None:
        payload = {
            "schema": "linkerhand_g20_gui_waypoints_v1",
            "joint_space": "sdk_range_0_255",
            "waypoints": self.waypoints,
            "trajectory_waypoints": [
                {
                    **({} if index == 0 else {"duration": item["suggested_duration"]}),
                    "pose": item["command"],
                }
                for index, item in enumerate(self.waypoints)
            ],
        }
        (self.path / "robot" / "waypoints.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )

    def finalize(self, *, status: str = "complete") -> None:
        if not self.samples_file.closed:
            self.samples_file.close()
        if self.waypoints:
            self._write_waypoints()
        metadata = {
            "schema": "linkerhand_grouped_action_v1",
            "group_index": self.index,
            "status": status,
            "created_at": datetime.fromtimestamp(self.started_wall).isoformat(timespec="seconds"),
            "human_samples": self.sample_count,
            "human_fresh_samples": self.fresh_count,
            "human_duration_seconds": (
                None
                if self.human_stopped_mono is None
                else self.human_stopped_mono - self.human_started_mono
            ),
            "repetition_markers": self.markers,
            "human_takes": self.human_takes,
            "robot_waypoints": len(self.waypoints),
        }
        (self.path / "group.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )


def _optional_points(value) -> Optional[list]:
    if value is None:
        return None
    array = np.asarray(value, dtype=float)
    if not np.all(np.isfinite(array)):
        return None
    return array.tolist()


def draw_hand_overlay(frame: np.ndarray, landmarks_px, *, fresh: bool) -> np.ndarray:
    """Draw the MediaPipe skeleton on a preview copy, leaving raw data untouched."""
    import cv2

    preview = np.asarray(frame).copy()
    if landmarks_px is None:
        return preview
    points = np.asarray(landmarks_px, dtype=float)
    if points.shape != (21, 2) or not np.all(np.isfinite(points)):
        return preview
    height, width = preview.shape[:2]
    pixels = np.rint(points).astype(int)
    line_colour = (0, 255, 0) if fresh else (0, 215, 255)
    point_colour = (0, 80, 255) if fresh else (0, 165, 255)
    for start, end in HAND_CONNECTIONS:
        a = tuple(pixels[start])
        b = tuple(pixels[end])
        if (
            0 <= a[0] < width and 0 <= a[1] < height
            and 0 <= b[0] < width and 0 <= b[1] < height
        ):
            cv2.line(preview, a, b, line_colour, 2, cv2.LINE_AA)
    for x, y in pixels:
        if 0 <= x < width and 0 <= y < height:
            cv2.circle(preview, (int(x), int(y)), 4, point_colour, -1, cv2.LINE_AA)
            cv2.circle(preview, (int(x), int(y)), 5, (255, 255, 255), 1, cv2.LINE_AA)
    return preview


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data/action_groups"))
    parser.add_argument("--session-name", default="g20_action_library")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--side", choices=("right", "left"), default="right")
    parser.add_argument("--mediapipe-camera", type=int, default=2)
    parser.add_argument("--robot-camera", type=int, default=0)
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument("--camera-fps", type=float, default=30.0)
    parser.add_argument("--camera-fourcc", default="MJPG")
    parser.add_argument("--jpeg-quality", type=int, default=92)
    parser.add_argument("--min-detection-confidence", type=float, default=0.75)
    parser.add_argument("--min-tracking-confidence", type=float, default=0.75)
    parser.add_argument("--min-hand-score", type=float, default=0.5)
    parser.add_argument("--ros-stale-seconds", type=float, default=0.5)
    parser.add_argument("--require-state", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--suggested-duration", type=float, default=0.5)
    parser.add_argument(
        "--minimal-overlay",
        action="store_true",
        help="hide the dynamic message and keyboard-help lines in the preview",
    )
    return parser.parse_args(argv)


def _open_robot_camera(args: argparse.Namespace):
    import cv2

    if args.robot_camera == args.mediapipe_camera:
        raise ValueError("--robot-camera and --mediapipe-camera must be different")
    backend = cv2.CAP_V4L2 if hasattr(cv2, "CAP_V4L2") else 0
    camera = cv2.VideoCapture(args.robot_camera, backend)
    if not camera.isOpened():
        raise RuntimeError(f"could not open robot camera {args.robot_camera}")
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, args.camera_width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, args.camera_height)
    camera.set(cv2.CAP_PROP_FPS, args.camera_fps)
    if len(args.camera_fourcc) == 4:
        camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*args.camera_fourcc))
    return camera


def _mosaic(left: np.ndarray, right: np.ndarray, status: list[str]) -> np.ndarray:
    import cv2

    height = 480
    def fit(frame):
        scale = height / frame.shape[0]
        return cv2.resize(frame, (int(round(frame.shape[1] * scale)), height))
    canvas = np.concatenate((fit(left), fit(right)), axis=1)
    for row, text in enumerate(status):
        cv2.putText(
            canvas, text, (12, 30 + row * 25), cv2.FONT_HERSHEY_SIMPLEX,
            0.62, (0, 255, 255), 2,
        )
    return canvas


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    if args.camera_fps <= 0 or args.suggested_duration <= 0:
        print("[group_recorder] fps and suggested duration must be positive", file=sys.stderr)
        return 2
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = args.output_dir / f"{stamp}_{args.session_name}"
    session_dir.mkdir(parents=True, exist_ok=False)
    (session_dir / "session.json").write_text(json.dumps({
        "schema": "linkerhand_grouped_action_session_v1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "side": args.side,
        "mediapipe_camera": args.mediapipe_camera,
        "robot_camera": args.robot_camera,
        "command_topic": f"/cb_{args.side}_hand_control_cmd",
        "state_topic": f"/cb_{args.side}_hand_state",
        "args": vars(args) | {"output_dir": str(args.output_dir)},
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    import cv2
    import rclpy

    rclpy.init(args=None)
    monitor = GuiCommandMonitor(args.side)
    source = None
    robot_camera = None
    current: Optional[GroupCapture] = None
    workflow = GroupWorkflow(args.start_index)
    latest_robot_frame = None
    last_hand_pixels = None
    message = "SPACE: start first group"
    try:
        source = MediaPipeHandSource(
            camera_index=args.mediapipe_camera,
            min_detection_confidence=args.min_detection_confidence,
            min_tracking_confidence=args.min_tracking_confidence,
            fps=args.camera_fps,
        )
        pipeline = HandPipeline(source, force_side=args.side, min_score=args.min_hand_score)
        robot_camera = _open_robot_camera(args)
        print(f"[group_recorder] session={session_dir}", flush=True)
        print("[group_recorder] SPACE group/robot/next; M human take start/stop; S robot snapshot; Q/E retry; X exit", flush=True)
        while True:
            monitor.spin()
            detection = source.read()
            processed = pipeline.process(detection)
            ok_robot, robot_frame = robot_camera.read()
            if ok_robot:
                latest_robot_frame = robot_frame.copy()
            human_frame = source.last_frame_bgr.copy()
            if workflow.phase == Phase.HUMAN_RECORDING and current is not None:
                current.add_human(human_frame, processed, source)

            robot_preview = (
                latest_robot_frame
                if latest_robot_frame is not None
                else np.zeros_like(human_frame)
            )
            fresh = bool(processed is not None and processed.detected and not processed.held)
            current_pixels = getattr(source, "last_landmarks_raw_px", None)
            if current_pixels is not None:
                last_hand_pixels = current_pixels
            human_preview = draw_hand_overlay(
                human_frame,
                current_pixels if fresh else last_hand_pixels,
                fresh=fresh,
            )
            status = [
                f"group={workflow.group_index:03d} phase={workflow.phase.value}",
                f"human_fresh={fresh} cmd={'yes' if monitor.command is not None else 'no'} state={'yes' if monitor.state is not None else 'no'}",
            ]
            if not args.minimal_overlay:
                status.extend((
                    message,
                    "SPACE=group/robot/next  M=start/stop take  S=snapshot  Q=redo human  E=redo robot  X/ESC=exit",
                ))
            cv2.imshow("human MediaPipe | robot waypoint", _mosaic(human_preview, robot_preview, status))
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("x"), ord("X"), 27):
                break
            if key in (ord("q"), ord("Q")):
                if current is None:
                    message = "Q ignored: no current group"
                else:
                    revision = current.reset_human()
                    workflow.redo_human()
                    message = f"redo HUMAN; previous attempt archived at {revision.name}"
                    print(f"[group_recorder] {message}", flush=True)
            if key in (ord("e"), ord("E")):
                if current is None:
                    message = "E ignored: no current group"
                elif workflow.phase != Phase.ROBOT:
                    message = "E ignored: finish human takes and enter ROBOT with SPACE first"
                else:
                    revision = current.reset_robot()
                    workflow.redo_robot()
                    message = f"redo ROBOT; previous waypoints archived at {revision.name}"
                    print(f"[group_recorder] {message}", flush=True)
            if key in (ord("m"), ord("M")):
                if current is None:
                    message = "M ignored: press SPACE to create a group first"
                else:
                    event = workflow.toggle_human_take()
                    if event.action == "start_take":
                        take_index = current.start_human_take()
                        message = f"HUMAN REC take {take_index}; press M to stop"
                    elif event.action == "stop_take":
                        take = current.finish_human_take()
                        message = f"saved human take {take['take_index']} frames={take['frames']}"
                    else:
                        message = "M ignored: human phase is finished for this group"
                    print(f"[group_recorder] {message}", flush=True)
            if key in (ord("s"), ord("S")):
                if workflow.phase != Phase.ROBOT or current is None:
                    message = "S ignored: first stop HUMAN with SPACE"
                elif latest_robot_frame is None:
                    message = "S failed: no robot camera frame"
                else:
                    try:
                        snapshot = monitor.snapshot(args.ros_stale_seconds, args.require_state)
                        image_path = current.add_robot_waypoint(
                            latest_robot_frame, snapshot,
                            suggested_duration=args.suggested_duration,
                        )
                        workflow.waypoint()
                        message = f"saved waypoint {workflow.robot_waypoints - 1}: {image_path.name}"
                        print(f"[group_recorder] {message} command={snapshot['command']}", flush=True)
                    except RuntimeError as exc:
                        message = f"S failed: {exc}"
                        print(f"[group_recorder] {message}", file=sys.stderr, flush=True)
            if key == ord(" "):
                event = workflow.space()
                if event.action == "start_group":
                    current = GroupCapture(session_dir, workflow.group_index, args.jpeg_quality)
                    message = f"group {workflow.group_index:03d} HUMAN READY; press M to record each take"
                elif event.action == "stop_human":
                    assert current is not None
                    current.stop_human()
                    message = "ROBOT CAPTURE; pose with official GUI, press S for each waypoint"
                elif event.action == "need_human_take":
                    message = "group not advanced: record at least one human take with M"
                elif event.action == "need_stop_take":
                    message = "human take is recording: press M to stop it before SPACE"
                elif event.action == "need_waypoint":
                    message = "group not closed: press S to save at least one robot waypoint"
                elif event.action == "finalize_and_start":
                    assert current is not None
                    current.finalize(status="complete")
                    completed = current.index
                    current = GroupCapture(session_dir, workflow.group_index, args.jpeg_quality)
                    message = f"saved group {completed:03d}; group {workflow.group_index:03d} HUMAN READY, press M"
                print(f"[group_recorder] {message}", flush=True)
        return 0
    except KeyboardInterrupt:
        return 130
    finally:
        if current is not None:
            status = "complete" if current.waypoints and workflow.phase == Phase.ROBOT else "incomplete"
            current.finalize(status=status)
        if source is not None:
            source.close()
        if robot_camera is not None:
            robot_camera.release()
        monitor.close()
        cv2.destroyAllWindows()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
