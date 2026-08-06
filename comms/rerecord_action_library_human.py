#!/usr/bin/env python3
"""Re-record selected human templates in an action library without touching robot motion.

The capture is staged in a new session first.  By default every primitive is
recorded; ``--action-id`` can limit capture and installation to one or more
primitives.  After every selected primitive has enough takes,
leave-one-take-out validation must pass before its human NPY files and threshold
are replaced.  The manifest and former human templates are archived; robot
trajectories are never written by this process.  This module does not import ROS
and cannot publish hardware commands.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np

from src.comms.action_library import (
    ActionLibrary,
    FEATURE_PROFILE_FULL,
    dtw_distance,
    landmark_feature,
)
from src.comms.group_action_recorder import GroupCapture, draw_hand_overlay
from src.comms.import_action_group import _automatic_threshold
from src.perception.mediapipe_source import MediaPipeHandSource
from src.perception.pipeline import HandPipeline


ACTION_GUIDANCE = {
    1: "four fingers full close",
    2: "thumb fold inward",
    3: "move thumb to the little finger, then move it back to the right",
    4: "coordinated finger transition",
    5: "index leads, then four fingers close",
}


@dataclass(frozen=True)
class Target:
    primitive_id: int
    name: str
    record: dict[str, Any]


@dataclass(frozen=True)
class ValidationResult:
    correct: int
    total: int
    thresholds: dict[int, float]
    misses: tuple[str, ...]

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total else 0.0


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--camera-index", type=int, default=2)
    parser.add_argument("--side", choices=("right", "left"), default="right")
    parser.add_argument(
        "--action-id",
        type=int,
        action="append",
        default=None,
        help=(
            "only rerecord this action ID; repeat the option for multiple "
            "actions (default: all actions)"
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/human_rerecordings"))
    parser.add_argument("--session-name", default="core_actions_human_rerecord")
    parser.add_argument("--takes-per-action", type=int, default=5)
    parser.add_argument("--min-take-frames", type=int, default=8)
    parser.add_argument("--camera-fps", type=float, default=30.0)
    parser.add_argument("--jpeg-quality", type=int, default=92)
    parser.add_argument("--min-detection-confidence", type=float, default=0.75)
    parser.add_argument("--min-tracking-confidence", type=float, default=0.75)
    parser.add_argument("--min-hand-score", type=float, default=0.5)
    parser.add_argument("--validation-margin", type=float, default=0.015)
    parser.add_argument("--minimum-accuracy", type=float, default=1.0)
    parser.add_argument(
        "--install",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="install validated human templates into the library; robot NPY files are untouched",
    )
    return parser.parse_args(argv)


def _targets(
    manifest: dict[str, Any], action_ids: Optional[list[int]] = None
) -> list[Target]:
    targets = [
        Target(int(item["id"]), str(item["name"]), item)
        for item in manifest.get("primitives", [])
    ]
    targets.sort(key=lambda item: item.primitive_id)
    if not targets:
        raise ValueError("action library contains no primitives")
    if action_ids is not None:
        requested = set(action_ids)
        available = {target.primitive_id for target in targets}
        missing = sorted(requested - available)
        if missing:
            raise ValueError(
                f"unknown action IDs {missing}; available={sorted(available)}"
            )
        targets = [target for target in targets if target.primitive_id in requested]
    return targets


def _fresh_takes(group: Path, *, min_frames: int) -> list[np.ndarray]:
    metadata = json.loads((group / "group.json").read_text(encoding="utf-8"))
    rows = [
        json.loads(line)
        for line in (group / "human" / "samples.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    takes: list[np.ndarray] = []
    for item in metadata.get("human_takes", []):
        start = int(item["start_sample"])
        end = int(item["end_sample"])
        frames = []
        for row in rows[start:end]:
            value = row.get("landmarks_hand_base")
            if not row.get("fresh") or value is None:
                continue
            landmarks = np.asarray(value, dtype=np.float32)
            if landmarks.shape == (21, 3) and np.all(np.isfinite(landmarks)):
                frames.append(landmarks)
        if len(frames) < min_frames:
            raise ValueError(
                f"{group.name} take {item.get('take_index')} has only "
                f"{len(frames)} fresh frames; need {min_frames}"
            )
        takes.append(np.stack(frames))
    if len(takes) < 2:
        raise ValueError(f"{group.name} needs at least two completed human takes")
    return takes


def _fresh_counts(capture: GroupCapture) -> list[int]:
    """Return fresh MediaPipe-frame counts for the current in-progress takes."""
    capture.samples_file.flush()
    rows = [
        json.loads(line)
        for line in capture.samples_path.read_text(encoding="utf-8").splitlines()
    ]
    return [
        sum(
            bool(row.get("fresh") and row.get("landmarks_hand_base") is not None)
            for row in rows[int(item["start_sample"]):int(item["end_sample"])]
        )
        for item in capture.human_takes
    ]


def _complete_validation_takes(
    library_path: Path,
    manifest: dict[str, Any],
    replacements: dict[int, list[np.ndarray]],
) -> dict[int, list[np.ndarray]]:
    """Combine staged replacements with existing takes for untouched actions."""
    takes_by_id = dict(replacements)
    for record in manifest.get("primitives", []):
        primitive_id = int(record["id"])
        if primitive_id in takes_by_id:
            continue
        paths = list(record.get("human_templates", []))
        if not paths:
            raise ValueError(f"primitive {primitive_id} has no human templates")
        takes_by_id[primitive_id] = [
            np.load(library_path / path, allow_pickle=False) for path in paths
        ]
    return takes_by_id


def validate_replacements(
    takes_by_id: dict[int, list[np.ndarray]],
    *,
    feature_profile: str,
    margin: float,
) -> ValidationResult:
    """Strict leave-one-take-out validation for a complete replacement set."""
    features: dict[int, list[np.ndarray]] = {}
    thresholds: dict[int, float] = {}
    for primitive_id, takes in takes_by_id.items():
        if len(takes) < 2:
            raise ValueError(f"primitive {primitive_id} needs at least two takes")
        features[primitive_id] = [
            np.stack([
                landmark_feature(frame, feature_profile=feature_profile)
                for frame in take
            ])
            for take in takes
        ]
        thresholds[primitive_id] = _automatic_threshold(
            takes, feature_profile=feature_profile
        )

    correct = 0
    total = 0
    misses: list[str] = []
    for expected_id, expected_takes in features.items():
        for held_index, held in enumerate(expected_takes):
            scores: list[tuple[float, int]] = []
            for candidate_id, candidate_takes in features.items():
                candidates = [
                    candidate
                    for index, candidate in enumerate(candidate_takes)
                    if candidate_id != expected_id or index != held_index
                ]
                if candidates:
                    scores.append((
                        min(dtw_distance(held, candidate) for candidate in candidates),
                        candidate_id,
                    ))
            scores.sort()
            if not scores:
                continue
            best_distance, best_id = scores[0]
            second = scores[1][0] if len(scores) > 1 else np.inf
            predicted = (
                best_id
                if best_distance <= thresholds[best_id]
                and second - best_distance >= max(0.0, float(margin))
                else -1
            )
            total += 1
            if predicted == expected_id:
                correct += 1
            else:
                misses.append(
                    f"expected={expected_id} take={held_index} predicted={predicted} "
                    f"best={best_distance:.4f} second={second:.4f}"
                )
    return ValidationResult(correct, total, thresholds, tuple(misses))


def _unique_archive(root: Path, label: str) -> Path:
    base = root / "archive" / label
    candidate = base
    index = 1
    while candidate.exists():
        candidate = root / "archive" / f"{label}_{index:02d}"
        index += 1
    candidate.mkdir(parents=True)
    return candidate


def install_replacements(
    library_path: Path,
    manifest: dict[str, Any],
    targets: list[Target],
    takes_by_id: dict[int, list[np.ndarray]],
    validation: ValidationResult,
    *,
    session_dir: Path,
) -> Path:
    """Replace only human templates and manifest thresholds, with an archive."""
    library_path = library_path.resolve()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive = _unique_archive(library_path, f"{stamp}_before_human_rerecord")
    shutil.copy2(library_path / "manifest.json", archive / "manifest.json")

    # Resolve every target before changing anything.
    primitive_dirs: dict[int, Path] = {}
    for target in targets:
        primitive_dir = (library_path / target.record["robot_trajectory"]).parent
        if not primitive_dir.is_dir():
            raise FileNotFoundError(f"missing primitive directory {primitive_dir}")
        primitive_dirs[target.primitive_id] = primitive_dir

    updated_at = datetime.now().isoformat(timespec="seconds")
    for target in targets:
        primitive_dir = primitive_dirs[target.primitive_id]
        primitive_archive = archive / primitive_dir.name
        primitive_archive.mkdir(parents=True)
        for old in sorted(primitive_dir.glob("human_take_*.npy")):
            shutil.move(str(old), str(primitive_archive / old.name))

        relative_templates = []
        for take_index, take in enumerate(takes_by_id[target.primitive_id]):
            path = primitive_dir / f"human_take_{take_index:03d}.npy"
            np.save(path, np.asarray(take, dtype=np.float32), allow_pickle=False)
            relative_templates.append(str(path.relative_to(library_path)))
        target.record["human_templates"] = relative_templates
        target.record["threshold"] = validation.thresholds[target.primitive_id]
        target.record["human_templates_updated_at"] = updated_at
        target.record["human_source_session"] = str(session_dir.resolve())
        target.record["human_source_group"] = f"group_{target.primitive_id:03d}"

    manifest["human_templates_updated_at"] = updated_at
    manifest["human_source_session"] = str(session_dir.resolve())
    temporary = library_path / "manifest.json.tmp"
    temporary.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(library_path / "manifest.json")
    ActionLibrary.load(library_path)
    return archive


def _finalize_capture(capture: GroupCapture, target: Target) -> None:
    capture.stop_human()
    capture.finalize(status="human_only_staged")
    metadata_path = capture.path / "group.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.update({
        "library_id": target.primitive_id,
        "action_name": target.name,
        "capture_mode": "human_only_preserve_robot_trajectory",
    })
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _overlay(frame: np.ndarray, *, target: Target, position: int, total: int,
             recording: bool, takes: int, required: int, message: str) -> np.ndarray:
    import cv2

    preview = frame.copy()
    colour = (0, 0, 255) if recording else (0, 220, 255)
    rows = (
        f"HUMAN ONLY  action {position}/{total}: {target.primitive_id} {target.name}",
        f"Do: {ACTION_GUIDANCE.get(target.primitive_id, target.name)}",
        f"{'RECORDING' if recording else 'READY'}  takes={takes}/{required}",
        message,
        "M=start/stop take  SPACE=accept/next  Q=redo action  X/ESC=abort",
        "Robot trajectories are never written by this recorder",
    )
    for row, text in enumerate(rows):
        cv2.putText(
            preview, text, (12, 30 + row * 26), cv2.FONT_HERSHEY_SIMPLEX,
            0.58, colour if row == 2 else (255, 255, 255), 2 if row < 3 else 1,
        )
    return preview


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    if args.takes_per_action < 2 or args.min_take_frames < 2:
        print("[human_rerecord] need at least two takes and two frames", file=sys.stderr)
        return 2
    if not 0.0 <= args.minimum_accuracy <= 1.0 or args.camera_fps <= 0:
        print("[human_rerecord] invalid accuracy or camera FPS", file=sys.stderr)
        return 2

    library_path = args.library.resolve()
    try:
        manifest = json.loads((library_path / "manifest.json").read_text(encoding="utf-8"))
        library = ActionLibrary.load(library_path)
        targets = _targets(manifest, args.action_id)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"[human_rerecord] cannot load library: {exc}", file=sys.stderr)
        return 2

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = args.output_dir / f"{stamp}_{args.session_name}"
    session_dir.mkdir(parents=True, exist_ok=False)
    (session_dir / "session.json").write_text(json.dumps({
        "schema": "linkerhand_human_rerecord_session_v1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "library": str(library_path),
        "feature_profile": library.feature_profile,
        "side": args.side,
        "camera_index": args.camera_index,
        "takes_per_action": args.takes_per_action,
        "actions": [
            {"id": item.primitive_id, "name": item.name} for item in targets
        ],
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    source = None
    capture: Optional[GroupCapture] = None
    completed_groups: dict[int, Path] = {}
    target_index = 0
    recording = False
    message = "Press M, perform one complete action, then press M again"
    try:
        import cv2

        source = MediaPipeHandSource(
            camera_index=args.camera_index,
            min_detection_confidence=args.min_detection_confidence,
            min_tracking_confidence=args.min_tracking_confidence,
            fps=args.camera_fps,
        )
        pipeline = HandPipeline(
            source, force_side=args.side, min_score=args.min_hand_score
        )
        target = targets[target_index]
        capture = GroupCapture(session_dir, target.primitive_id, args.jpeg_quality)
        print(f"[human_rerecord] session={session_dir}", flush=True)
        print("[human_rerecord] no ROS publisher; robot trajectories are read-only", flush=True)
        while True:
            detection = source.read()
            processed = pipeline.process(detection)
            fresh = bool(processed is not None and processed.detected and not processed.held)
            raw_frame = source.last_frame_bgr.copy()
            if recording:
                capture.add_human(raw_frame, processed, source)
            preview = draw_hand_overlay(
                raw_frame,
                getattr(source, "last_landmarks_raw_px", None),
                fresh=fresh,
            )
            preview = _overlay(
                preview,
                target=target,
                position=target_index + 1,
                total=len(targets),
                recording=recording,
                takes=len(capture.human_takes),
                required=args.takes_per_action,
                message=message,
            )
            cv2.imshow("human-only action-library rerecord", preview)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("x"), ord("X"), 27):
                message = "aborted; staged session kept and library unchanged"
                print(f"[human_rerecord] {message}: {session_dir}", flush=True)
                return 130
            if key in (ord("q"), ord("Q")):
                if recording:
                    message = "press M to stop the active take before Q"
                else:
                    revision = capture.reset_human()
                    message = f"current action cleared; old attempt={revision.name}"
                    print(f"[human_rerecord] {message}", flush=True)
            if key in (ord("m"), ord("M")):
                if not recording:
                    take_index = capture.start_human_take()
                    recording = True
                    message = f"recording take {take_index}; press M when complete"
                else:
                    item = capture.finish_human_take()
                    recording = False
                    message = f"saved take {item['take_index']} frames={item['frames']}"
                print(f"[human_rerecord] action={target.primitive_id} {message}", flush=True)
            if key == ord(" "):
                if recording:
                    message = "press M to stop the active take first"
                    continue
                if len(capture.human_takes) < args.takes_per_action:
                    message = (
                        f"need {args.takes_per_action} takes; "
                        f"currently {len(capture.human_takes)}"
                    )
                    continue
                counts = _fresh_counts(capture)
                short = [
                    index for index, count in enumerate(counts)
                    if count < args.min_take_frames
                ]
                if short:
                    message = (
                        f"takes {short} have too few fresh frames {counts}; "
                        "press Q and rerecord this action"
                    )
                    print(f"[human_rerecord] {message}", file=sys.stderr, flush=True)
                    continue
                _finalize_capture(capture, target)
                completed_groups[target.primitive_id] = capture.path
                target_index += 1
                if target_index >= len(targets):
                    break
                target = targets[target_index]
                capture = GroupCapture(
                    session_dir, target.primitive_id, args.jpeg_quality
                )
                message = "next action ready; press M to record take 0"
                print(
                    f"[human_rerecord] next {target.primitive_id}:{target.name}",
                    flush=True,
                )

        takes_by_id = {
            target.primitive_id: _fresh_takes(
                completed_groups[target.primitive_id],
                min_frames=args.min_take_frames,
            )
            for target in targets
        }
        validation_takes = _complete_validation_takes(
            library_path, manifest, takes_by_id
        )
        validation = validate_replacements(
            validation_takes,
            feature_profile=library.feature_profile or FEATURE_PROFILE_FULL,
            margin=args.validation_margin,
        )
        report = {
            "schema": "linkerhand_human_rerecord_validation_v1",
            "feature_profile": library.feature_profile,
            "installed_action_ids": sorted(takes_by_id),
            "validation_action_ids": sorted(validation_takes),
            "correct": validation.correct,
            "total": validation.total,
            "accuracy": validation.accuracy,
            "thresholds": validation.thresholds,
            "misses": list(validation.misses),
        }
        (session_dir / "validation.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(
            f"[human_rerecord] validation={validation.correct}/{validation.total} "
            f"accuracy={validation.accuracy:.3%}",
            flush=True,
        )
        for miss in validation.misses:
            print(f"[human_rerecord] MISS {miss}", file=sys.stderr, flush=True)
        if validation.accuracy < args.minimum_accuracy:
            print(
                f"[human_rerecord] NOT INSTALLED: need accuracy "
                f">={args.minimum_accuracy:.1%}; staged data kept at {session_dir}",
                file=sys.stderr,
            )
            return 3
        if not args.install:
            print(f"[human_rerecord] validated staging kept at {session_dir}", flush=True)
            return 0
        archive = install_replacements(
            library_path,
            manifest,
            targets,
            takes_by_id,
            validation,
            session_dir=session_dir,
        )
        print(f"[human_rerecord] INSTALLED human templates; archive={archive}", flush=True)
        print("[human_rerecord] robot trajectories unchanged", flush=True)
        return 0
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"[human_rerecord] failed: {exc}", file=sys.stderr)
        print(f"[human_rerecord] staged data kept at {session_dir}", file=sys.stderr)
        return 2
    finally:
        if capture is not None and not capture.samples_file.closed:
            capture.samples_file.close()
        if source is not None:
            source.close()
        try:
            import cv2

            cv2.destroyAllWindows()
        except ImportError:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
