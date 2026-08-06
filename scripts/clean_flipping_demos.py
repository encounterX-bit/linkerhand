#!/usr/bin/env python3
"""Build a deterministic, atomic flipping dataset without touching raw demos.

Long teleoperation episodes are segmented at stable ArUco cube-face changes.
Only explicitly allowed face transitions are retained.  By default, only the
final stable transition in each source episode is considered: transient
back-and-forth face changes therefore cannot become extra demonstrations.  A
derived episode starts when the final source face becomes stable and ends at
the end of the stable destination-face run.  Optional context limits can trim
those runs further.  Image paths point back to the immutable source JPEGs, so
no images are copied.

The default mode is a read-only audit.  Pass ``--write`` to create the derived
dataset and its ``cleaning_report.json`` provenance report.
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


JOINT_DIM = 20
RESERVED_IDX = (11, 12, 13, 14)
ACTIVE_IDX = tuple(i for i in range(JOINT_DIM) if i not in RESERVED_IDX)
FLEXION_JOINTS = (1, 2, 3, 4, 16, 17, 18, 19)
DEFAULT_ROUTE = "0>4,1>4,2>5,3>2,4>2,5>3"


def all_face_changes(face_count: int = 6) -> set[tuple[int, int]]:
    """Return every directed transition between distinct cube faces."""
    if face_count < 2:
        raise ValueError("face_count must be at least two")
    return {
        (source, destination)
        for source in range(face_count)
        for destination in range(face_count)
        if source != destination
    }


@dataclass(frozen=True)
class FaceRun:
    face: int
    start_frame: int
    end_frame: int
    samples: int


@dataclass(frozen=True)
class Segment:
    source_episode: Path
    from_face: int
    to_face: int
    start_frame: int
    end_frame: int
    transition_left_frame: int
    transition_right_frame: int
    action_reversals: int
    thumb_route_reversals: int
    total_action_variation: float

    @property
    def frame_count(self) -> int:
        return self.end_frame - self.start_frame + 1


def parse_route(value: str) -> set[tuple[int, int]]:
    route: set[tuple[int, int]] = set()
    try:
        for item in value.split(","):
            left, right = item.strip().split(">", 1)
            transition = (int(left), int(right))
            if any(face < 0 for face in transition):
                raise ValueError
            route.add(transition)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "route must look like 0>4,1>4,2>5"
        ) from exc
    if not route:
        raise argparse.ArgumentTypeError("route cannot be empty")
    return route


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", type=Path, default=Path("data/act_demos"))
    ap.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/act_demos_clean/flipping_canonical_v1"),
    )
    ap.add_argument("--task-id", default="action_library_hybrid_demo_v1")
    ap.add_argument(
        "--output-task-id",
        default=None,
        help="task_id written to the derived session; default appends canonical_clean_v1",
    )
    ap.add_argument(
        "--include-session",
        action="append",
        default=[],
        help="raw session directory name to include; repeat for multiple sessions",
    )
    ap.add_argument("--route", type=parse_route, default=parse_route(DEFAULT_ROUTE))
    ap.add_argument(
        "--allow-any-face-change",
        action="store_true",
        help="accept all 30 directed changes among ArUco cube faces 0..5",
    )
    ap.add_argument("--aruco-dictionary", default="DICT_4X4_50")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--sample-stride", type=int, default=5)
    ap.add_argument("--smooth-radius", type=int, default=2)
    ap.add_argument(
        "--stable-samples",
        type=int,
        default=18,
        help=(
            "minimum sampled detections for a stable face run; with the "
            "defaults, 18 samples at stride 5 and 30 Hz is about 3 seconds"
        ),
    )
    ap.add_argument(
        "--selection",
        choices=("final-transition", "all-transitions"),
        default="final-transition",
        help=(
            "final-transition keeps at most one atomic demonstration per raw "
            "episode and rejects intermediate back-and-forth cube motion"
        ),
    )
    ap.add_argument(
        "--context-before",
        type=int,
        default=0,
        help=(
            "maximum frames before the face boundary; 0 keeps the complete "
            "stable source-face run"
        ),
    )
    ap.add_argument(
        "--context-after",
        type=int,
        default=0,
        help=(
            "maximum frames after the face boundary; 0 keeps the complete "
            "stable destination-face run"
        ),
    )
    ap.add_argument("--min-segment-frames", type=int, default=60)
    ap.add_argument("--max-segment-frames", type=int, default=0)
    ap.add_argument("--max-action-reversals", type=int, default=None)
    ap.add_argument("--max-thumb-route-reversals", type=int, default=None)
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    args.data_root = args.data_root.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    if args.allow_any_face_change:
        args.route = all_face_changes()
    if not args.data_root.is_dir():
        ap.error(f"--data-root does not exist: {args.data_root}")
    if (
        args.fps <= 0
        or args.sample_stride <= 0
        or args.smooth_radius < 0
        or args.stable_samples <= 0
        or args.context_before < 0
        or args.context_after < 0
        or args.min_segment_frames <= 0
        or args.max_segment_frames < 0
    ):
        ap.error("frame, timing, and smoothing values must be nonnegative")
    if args.max_action_reversals is not None and args.max_action_reversals < 0:
        ap.error("--max-action-reversals must be nonnegative")
    if (
        args.max_thumb_route_reversals is not None
        and args.max_thumb_route_reversals < 0
    ):
        ap.error("--max-thumb-route-reversals must be nonnegative")
    if args.overwrite and not args.write:
        ap.error("--overwrite requires --write")
    return args


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                yield row


def usable_rows(episode_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in read_jsonl(episode_dir / "samples.jsonl"):
        image_path = row.get("image_path")
        state = row.get("joint_pos")
        action = row.get("last_action")
        if not (
            image_path
            and (episode_dir / str(image_path)).is_file()
            and isinstance(state, list)
            and len(state) >= JOINT_DIM
            and isinstance(action, list)
            and len(action) >= JOINT_DIM
        ):
            continue
        rows.append(row)
    return rows


def source_task_id(episode_dir: Path, rows: Sequence[dict[str, Any]]) -> str:
    session_path = episode_dir.parent / "session.json"
    if session_path.is_file():
        try:
            value = json.loads(session_path.read_text(encoding="utf-8"))
            if value.get("task_id"):
                return str(value["task_id"])
        except (OSError, json.JSONDecodeError, AttributeError):
            pass
    return str(rows[0].get("task_id", "")) if rows else ""


def detector(dictionary_name: str) -> tuple[Any, Any]:
    import cv2

    dictionary_id = getattr(cv2.aruco, dictionary_name, None)
    if dictionary_id is None:
        raise RuntimeError(f"unknown OpenCV ArUco dictionary {dictionary_name!r}")
    return cv2, cv2.aruco.getPredefinedDictionary(dictionary_id)


def detect_face(image_path: Path, cv2: Any, dictionary: Any) -> int | None:
    image = cv2.imread(str(image_path))
    if image is None:
        return None
    _corners, ids, _rejected = cv2.aruco.detectMarkers(image, dictionary)
    if ids is None:
        return None
    groups = Counter(int(marker_id) // 4 for marker_id in ids.reshape(-1))
    face, marker_count = groups.most_common(1)[0]
    return face if marker_count >= 2 else None


def smooth_faces(
    faces: Sequence[int | None], *, radius: int
) -> list[int | None]:
    smoothed: list[int | None] = []
    for index in range(len(faces)):
        window = [
            value
            for value in faces[
                max(0, index - radius) : min(len(faces), index + radius + 1)
            ]
            if value is not None
        ]
        smoothed.append(Counter(window).most_common(1)[0][0] if window else None)
    return smoothed


def stable_face_runs(
    frame_indices: Sequence[int],
    faces: Sequence[int | None],
    *,
    minimum_samples: int,
) -> list[FaceRun]:
    if len(frame_indices) != len(faces):
        raise ValueError("frame indices and faces must have equal lengths")
    raw: list[FaceRun] = []
    for frame, face in zip(frame_indices, faces):
        if face is None:
            continue
        if not raw or raw[-1].face != face:
            raw.append(FaceRun(face, frame, frame, 1))
        else:
            previous = raw[-1]
            raw[-1] = FaceRun(
                previous.face,
                previous.start_frame,
                frame,
                previous.samples + 1,
            )
    stable = [run for run in raw if run.samples >= minimum_samples]
    collapsed: list[FaceRun] = []
    for run in stable:
        if collapsed and collapsed[-1].face == run.face:
            previous = collapsed[-1]
            collapsed[-1] = FaceRun(
                previous.face,
                previous.start_frame,
                run.end_frame,
                previous.samples + run.samples,
            )
        else:
            collapsed.append(run)
    return collapsed


def count_smoothed_reversals(values: np.ndarray) -> int:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if len(values) < 2:
        return 0
    window = min(15, len(values))
    left = window // 2
    right = window - 1 - left
    padded = np.pad(values, (left, right), mode="edge")
    smoothed = np.convolve(padded, np.ones(window) / window, mode="valid")
    delta = np.diff(smoothed)
    direction = np.where(delta > 0.35, 1, np.where(delta < -0.35, -1, 0))
    runs: list[int] = []
    for value in direction[direction != 0]:
        sign = int(value)
        if not runs or sign != runs[-1]:
            runs.append(sign)
    return max(0, len(runs) - 1)


def segment_metrics(rows: Sequence[dict[str, Any]]) -> tuple[int, int, float]:
    actions = np.asarray(
        [row["last_action"][:JOINT_DIM] for row in rows], dtype=np.float32
    )
    closure = 255.0 - actions[:, list(FLEXION_JOINTS)].mean(axis=1)
    action_reversals = count_smoothed_reversals(closure)
    thumb_reversals = sum(
        count_smoothed_reversals(actions[:, joint]) for joint in (5, 10)
    )
    total_variation = float(
        np.abs(np.diff(actions[:, list(ACTIVE_IDX)], axis=0)).sum()
    )
    return action_reversals, thumb_reversals, total_variation


def candidate_segments(
    episode_dir: Path,
    rows: Sequence[dict[str, Any]],
    runs: Sequence[FaceRun],
    *,
    context_before: int,
    context_after: int,
    selection: str = "all-transitions",
) -> list[Segment]:
    segments: list[Segment] = []
    pairs = list(zip(runs, runs[1:]))
    if selection == "final-transition":
        pairs = pairs[-1:]
    elif selection != "all-transitions":
        raise ValueError(f"unknown selection mode: {selection}")
    for left, right in pairs:
        if left.face == right.face:
            continue
        start = (
            left.start_frame
            if context_before == 0
            else max(left.start_frame, left.end_frame - context_before)
        )
        end = (
            right.end_frame
            if context_after == 0
            else min(right.end_frame, right.start_frame + context_after)
        )
        if end < start:
            continue
        action_reversals, thumb_reversals, variation = segment_metrics(
            rows[start : end + 1]
        )
        segments.append(
            Segment(
                episode_dir,
                left.face,
                right.face,
                start,
                end,
                left.end_frame,
                right.start_frame,
                action_reversals,
                thumb_reversals,
                variation,
            )
        )
    return segments


def rejection_reason(segment: Segment, args: argparse.Namespace) -> str | None:
    if (segment.from_face, segment.to_face) not in args.route:
        return "noncanonical_transition"
    if segment.frame_count < args.min_segment_frames:
        return "segment_too_short"
    if args.max_segment_frames and segment.frame_count > args.max_segment_frames:
        return "segment_too_long"
    if (
        args.max_action_reversals is not None
        and segment.action_reversals > args.max_action_reversals
    ):
        return "too_many_action_reversals"
    if (
        args.max_thumb_route_reversals is not None
        and segment.thumb_route_reversals > args.max_thumb_route_reversals
    ):
        return "too_many_thumb_route_reversals"
    return None


def segment_record(segment: Segment, reason: str | None) -> dict[str, Any]:
    return {
        "source_episode": str(segment.source_episode),
        "transition": f"{segment.from_face}>{segment.to_face}",
        "from_face": segment.from_face,
        "to_face": segment.to_face,
        "start_frame": segment.start_frame,
        "end_frame": segment.end_frame,
        "frames": segment.frame_count,
        "transition_left_frame": segment.transition_left_frame,
        "transition_right_frame": segment.transition_right_frame,
        "action_reversals": segment.action_reversals,
        "thumb_route_reversals": segment.thumb_route_reversals,
        "total_action_variation": segment.total_action_variation,
        "status": "kept" if reason is None else "rejected",
        "reason": reason,
    }


def write_segment(
    output_root: Path,
    episode_index: int,
    segment: Segment,
    rows: Sequence[dict[str, Any]],
) -> None:
    destination = output_root / f"episode_{episode_index:03d}"
    destination.mkdir(parents=True, exist_ok=False)
    selected = rows[segment.start_frame : segment.end_frame + 1]
    first_elapsed = float(selected[0].get("elapsed", 0.0) or 0.0)
    with (destination / "samples.jsonl").open("w", encoding="utf-8") as stream:
        for new_index, source_row in enumerate(selected):
            row = dict(source_row)
            source_image = (
                segment.source_episode / str(source_row["image_path"])
            ).resolve()
            row.update({
                "index": new_index,
                "episode": episode_index,
                "elapsed": max(
                    0.0,
                    float(source_row.get("elapsed", first_elapsed) or first_elapsed)
                    - first_elapsed,
                ),
                "image_path": str(source_image),
                "source_episode": str(segment.source_episode),
                "source_frame_index": segment.start_frame + new_index,
                "clean_transition": f"{segment.from_face}>{segment.to_face}",
            })
            stream.write(
                json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            )
    metadata = segment_record(segment, None)
    metadata["derived_episode_index"] = episode_index
    (destination / "episode.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    cv2, dictionary = detector(args.aruco_dictionary)
    kept: list[tuple[Segment, list[dict[str, Any]]]] = []
    records: list[dict[str, Any]] = []
    transition_counts: Counter[str] = Counter()

    for samples_path in sorted(args.data_root.glob("*/episode_*/samples.jsonl")):
        episode_dir = samples_path.parent.resolve()
        if (
            args.include_session
            and episode_dir.parent.name not in args.include_session
        ):
            continue
        rows = usable_rows(episode_dir)
        if not rows or source_task_id(episode_dir, rows) != args.task_id:
            continue
        indices = list(range(0, len(rows), args.sample_stride))
        if indices[-1] != len(rows) - 1:
            indices.append(len(rows) - 1)
        faces = [
            detect_face(episode_dir / str(rows[index]["image_path"]), cv2, dictionary)
            for index in indices
        ]
        runs = stable_face_runs(
            indices,
            smooth_faces(faces, radius=args.smooth_radius),
            minimum_samples=args.stable_samples,
        )
        for segment in candidate_segments(
            episode_dir,
            rows,
            runs,
            context_before=args.context_before,
            context_after=args.context_after,
            selection=args.selection,
        ):
            reason = rejection_reason(segment, args)
            record = segment_record(segment, reason)
            records.append(record)
            transition_counts[record["transition"]] += 1
            if reason is None:
                kept.append((segment, rows))
                print(
                    f"[keep] {record['transition']} frames={segment.frame_count} "
                    f"action_rev={segment.action_reversals} "
                    f"thumb_rev={segment.thumb_route_reversals} "
                    f"source={episode_dir.parent.name}/{episode_dir.name}"
                )
            else:
                print(
                    f"[drop] {record['transition']} reason={reason} "
                    f"source={episode_dir.parent.name}/{episode_dir.name}"
                )

    report = {
        "schema": "linkerhand_flipping_clean_v1",
        "source_root": str(args.data_root),
        "output_root": str(args.output_root),
        "task_id": args.task_id,
        "output_task_id": (
            args.output_task_id
            or f"{args.task_id}_canonical_clean_v1"
        ),
        "include_sessions": list(args.include_session),
        "route": [f"{left}>{right}" for left, right in sorted(args.route)],
        "aruco_dictionary": args.aruco_dictionary,
        "fps": args.fps,
        "sample_stride": args.sample_stride,
        "smooth_radius": args.smooth_radius,
        "stable_samples": args.stable_samples,
        "selection": args.selection,
        "context_before": args.context_before,
        "context_after": args.context_after,
        "max_action_reversals": args.max_action_reversals,
        "max_thumb_route_reversals": args.max_thumb_route_reversals,
        "candidate_segments": len(records),
        "kept_segments": len(kept),
        "kept_frames": sum(segment.frame_count for segment, _rows in kept),
        "transition_counts": dict(sorted(transition_counts.items())),
        "segments": records,
    }
    print(
        f"[summary] candidates={len(records)} kept={len(kept)} "
        f"frames={report['kept_frames']} write={args.write}",
        flush=True,
    )
    if not kept:
        raise RuntimeError("cleaning rules retained no transition segments")
    if not args.write:
        return 0
    if args.output_root.exists():
        if not args.overwrite:
            raise RuntimeError(
                f"output already exists: {args.output_root}; pass --overwrite"
            )
        shutil.rmtree(args.output_root)
    args.output_root.mkdir(parents=True)
    session = {
        "task_id": report["output_task_id"],
        "source_task_id": args.task_id,
        "source_root": str(args.data_root),
        "derived": True,
        "route": report["route"],
    }
    (args.output_root / "session.json").write_text(
        json.dumps(session, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    for episode_index, (segment, rows) in enumerate(kept):
        write_segment(args.output_root, episode_index, segment, rows)
    (args.output_root / "cleaning_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"[write] clean dataset: {args.output_root}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
    except RuntimeError as exc:
        print(f"error: {exc}")
        raise SystemExit(2)
