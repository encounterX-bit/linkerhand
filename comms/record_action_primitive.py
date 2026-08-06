#!/usr/bin/env python3
"""Record one MediaPipe template and pair it with a hardcoded G20 trajectory.

This command never imports ROS and never publishes a hardware command.  Press
SPACE to start a take, perform the short gesture, then press SPACE to save it.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from src.comms.action_library import ActionLibrary, interpolate_waypoints
from src.perception.mediapipe_source import MediaPipeHandSource
from src.perception.pipeline import HandPipeline


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--primitive-id", type=int, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument(
        "--waypoints",
        type=Path,
        help="JSON list of {pose:[20], duration:seconds}; required for a new primitive",
    )
    parser.add_argument("--replace-trajectory", action="store_true")
    parser.add_argument("--threshold", type=float, default=0.18)
    parser.add_argument("--cooldown-frames", type=int, default=10)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--camera-index", type=int, default=2)
    parser.add_argument("--side", choices=("right", "left"), default="right")
    parser.add_argument("--min-detection-confidence", type=float, default=0.75)
    parser.add_argument("--min-tracking-confidence", type=float, default=0.75)
    parser.add_argument("--min-hand-score", type=float, default=0.5)
    parser.add_argument(
        "--auto-seconds",
        type=float,
        default=0.0,
        help="headless mode: record this many seconds after the first fresh frame",
    )
    parser.add_argument("--min-frames", type=int, default=6)
    return parser.parse_args(argv)


def _read_manifest(root: Path, fps: float) -> dict:
    path = root / "manifest.json"
    if not path.is_file():
        return {
            "schema": ActionLibrary.SCHEMA,
            "hand_model": "g20_palm_touch",
            "joint_space": "sdk_range_0_255",
            "fps": fps,
            "primitives": [],
        }
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != ActionLibrary.SCHEMA:
        raise ValueError(f"unsupported manifest schema {payload.get('schema')!r}")
    return payload


def _upsert_primitive(args: argparse.Namespace, manifest: dict) -> tuple[dict, Path]:
    records = manifest.setdefault("primitives", [])
    record = next((item for item in records if int(item["id"]) == args.primitive_id), None)
    folder = args.library / f"primitive_{args.primitive_id:03d}_{args.name}"
    folder.mkdir(parents=True, exist_ok=True)
    relative_trajectory = str((folder / "robot_trajectory.npy").relative_to(args.library))
    if record is None:
        if args.waypoints is None:
            raise ValueError("--waypoints is required when creating a primitive")
        record = {
            "id": args.primitive_id,
            "name": args.name,
            "robot_trajectory": relative_trajectory,
            "human_templates": [],
            "threshold": args.threshold,
            "interruptible": False,
            "cooldown_frames": args.cooldown_frames,
        }
        records.append(record)
    elif str(record["name"]) != args.name:
        raise ValueError(
            f"primitive {args.primitive_id} already exists as {record['name']!r}, not {args.name!r}"
        )
    if args.waypoints is not None and (record is None or args.replace_trajectory or not (args.library / record["robot_trajectory"]).is_file()):
        waypoints = json.loads(args.waypoints.read_text(encoding="utf-8"))
        trajectory = interpolate_waypoints(waypoints, fps=args.fps)
        np.save(folder / "robot_trajectory.npy", trajectory, allow_pickle=False)
        record["robot_trajectory"] = relative_trajectory
    records.sort(key=lambda item: int(item["id"]))
    return record, folder


def _draw_status(frame: np.ndarray, text: str) -> None:
    import cv2

    cv2.putText(frame, text, (15, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)
    cv2.putText(frame, "SPACE start/stop  Q/ESC cancel", (15, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)


def capture(args: argparse.Namespace) -> np.ndarray:
    import cv2

    source = MediaPipeHandSource(
        camera_index=args.camera_index,
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
        fps=args.fps,
    )
    pipeline = HandPipeline(source, force_side=args.side, min_score=args.min_hand_score)
    recording = False
    started = 0.0
    landmarks: list[np.ndarray] = []
    try:
        while True:
            detection = source.read()
            processed = pipeline.process(detection)
            fresh = processed is not None and processed.detected and not processed.held
            if args.auto_seconds > 0 and fresh and not recording:
                recording = True
                started = time.monotonic()
            if recording and fresh:
                landmarks.append(np.asarray(processed.landmarks, dtype=np.float32).copy())
            if args.auto_seconds > 0 and recording and time.monotonic() - started >= args.auto_seconds:
                break

            frame = source.last_frame_bgr.copy()
            state = f"REC {len(landmarks)} frames" if recording else "READY"
            if not fresh:
                state += " | NO FRESH HAND"
            _draw_status(frame, state)
            if args.auto_seconds <= 0:
                cv2.imshow("record action primitive", frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    raise KeyboardInterrupt
                if key == ord(" "):
                    if recording:
                        break
                    landmarks.clear()
                    recording = True
                    started = time.monotonic()
    finally:
        source.close()
        if args.auto_seconds <= 0:
            cv2.destroyAllWindows()
    if len(landmarks) < args.min_frames:
        raise ValueError(f"take has only {len(landmarks)} fresh frames; need at least {args.min_frames}")
    return np.stack(landmarks).astype(np.float32)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.library.mkdir(parents=True, exist_ok=True)
    try:
        manifest = _read_manifest(args.library, args.fps)
        record, folder = _upsert_primitive(args, manifest)
        print(f"[record] primitive={args.primitive_id}:{args.name}; perform gesture", flush=True)
        template = capture(args)
        take_index = len(record["human_templates"])
        path = folder / f"human_take_{take_index:03d}.npy"
        np.save(path, template, allow_pickle=False)
        record["human_templates"].append(str(path.relative_to(args.library)))
        (args.library / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print(f"[record] saved {path} shape={template.shape}", flush=True)
        return 0
    except KeyboardInterrupt:
        print("[record] cancelled", file=sys.stderr)
        return 130
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"[record] {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
