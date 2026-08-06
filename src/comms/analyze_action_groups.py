#!/usr/bin/env python3
"""Summarize grouped captures and extract candidate MediaPipe takes offline."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from src.comms.action_library import landmark_feature


def motion_segments(
    features: np.ndarray,
    *,
    threshold: float = 0.018,
    pause_frames: int = 10,
    pad_frames: int = 3,
    min_frames: int = 6,
) -> list[tuple[int, int]]:
    """Return half-open candidate motion intervals in feature-frame indices."""
    values = np.asarray(features, dtype=np.float32)
    if values.ndim != 2 or len(values) < 2:
        return []
    energy = np.sqrt(np.mean(np.diff(values, axis=0) ** 2, axis=1))
    moving = np.flatnonzero(energy >= float(threshold)) + 1
    if not len(moving):
        return []
    groups: list[list[int]] = [[int(moving[0])]]
    for index in moving[1:]:
        index = int(index)
        if index - groups[-1][-1] <= max(1, int(pause_frames)):
            groups[-1].append(index)
        else:
            groups.append([index])
    segments = []
    for group in groups:
        start = max(0, group[0] - max(0, int(pad_frames)))
        end = min(len(values), group[-1] + max(0, int(pad_frames)) + 1)
        if end - start >= max(2, int(min_frames)):
            segments.append((start, end))
    return segments


def marker_segments(
    sample_indices: Sequence[int], markers: Sequence[dict[str, Any]], min_frames: int
) -> list[tuple[int, int]]:
    """Map explicit raw-sample boundaries to half-open fresh-frame intervals."""
    if not sample_indices:
        return []
    raw_boundaries = [int(item["sample_index"]) for item in markers]
    boundaries = [0]
    for raw in raw_boundaries:
        boundaries.append(int(np.searchsorted(sample_indices, raw, side="left")))
    boundaries.append(len(sample_indices))
    boundaries = sorted(set(max(0, min(len(sample_indices), value)) for value in boundaries))
    return [
        (start, end)
        for start, end in zip(boundaries, boundaries[1:])
        if end - start >= max(2, int(min_frames))
    ]


def _load_fresh_samples(group: Path) -> tuple[list[int], np.ndarray]:
    indices: list[int] = []
    landmarks: list[np.ndarray] = []
    samples_path = group / "human" / "samples.jsonl"
    for line in samples_path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        value = row.get("landmarks_hand_base")
        if not row.get("fresh") or value is None:
            continue
        array = np.asarray(value, dtype=np.float32)
        if array.shape == (21, 3) and np.all(np.isfinite(array)):
            indices.append(int(row["index"]))
            landmarks.append(array)
    if not landmarks:
        return indices, np.empty((0, 21, 3), dtype=np.float32)
    return indices, np.stack(landmarks)


def analyze_group(group: Path, args: argparse.Namespace) -> dict[str, Any]:
    metadata = json.loads((group / "group.json").read_text(encoding="utf-8"))
    sample_indices, landmarks = _load_fresh_samples(group)
    features = (
        np.stack([landmark_feature(frame) for frame in landmarks])
        if len(landmarks)
        else np.empty((0, 83), dtype=np.float32)
    )
    markers = metadata.get("repetition_markers", [])
    if markers:
        segments = marker_segments(sample_indices, markers, args.min_segment_frames)
        method = "manual_markers"
    else:
        segments = motion_segments(
            features,
            threshold=args.motion_threshold,
            pause_frames=args.pause_frames,
            pad_frames=args.pad_frames,
            min_frames=args.min_segment_frames,
        )
        method = "motion_energy"

    analysis_dir = group / "analysis"
    analysis_dir.mkdir(exist_ok=True)
    takes = []
    for take_index, (start, end) in enumerate(segments):
        path = analysis_dir / f"human_take_{take_index:03d}.npy"
        np.save(path, landmarks[start:end], allow_pickle=False)
        takes.append({
            "take_index": take_index,
            "path": str(path.relative_to(group)),
            "fresh_frame_start": start,
            "fresh_frame_end": end,
            "source_sample_start": sample_indices[start],
            "source_sample_end": sample_indices[end - 1],
            "frames": end - start,
        })

    waypoint_path = group / "robot" / "waypoints.json"
    robot_waypoints = 0
    if waypoint_path.is_file():
        waypoint_payload = json.loads(waypoint_path.read_text(encoding="utf-8"))
        trajectory = waypoint_payload.get("trajectory_waypoints", [])
        robot_waypoints = len(trajectory)
        (analysis_dir / "trajectory_waypoints.json").write_text(
            json.dumps(trajectory, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    result = {
        "group": group.name,
        "status": metadata.get("status"),
        "human_samples": int(metadata.get("human_samples", 0)),
        "human_fresh_samples": len(landmarks),
        "segmentation_method": method,
        "candidate_takes": takes,
        "robot_waypoints": robot_waypoints,
        "ready": bool(takes and robot_waypoints >= 1),
    }
    (analysis_dir / "analysis.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--motion-threshold", type=float, default=0.018)
    parser.add_argument("--pause-frames", type=int, default=10)
    parser.add_argument("--pad-frames", type=int, default=3)
    parser.add_argument("--min-segment-frames", type=int, default=6)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    groups = sorted(path for path in args.session.glob("group_*" ) if path.is_dir())
    if not groups:
        print(f"[analyze_groups] no group directories under {args.session}", file=sys.stderr)
        return 2
    results = []
    for group in groups:
        try:
            result = analyze_group(group, args)
            results.append(result)
            print(
                f"[analyze_groups] {group.name}: fresh={result['human_fresh_samples']} "
                f"takes={len(result['candidate_takes'])} waypoints={result['robot_waypoints']} "
                f"ready={result['ready']}",
                flush=True,
            )
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            results.append({"group": group.name, "ready": False, "error": str(exc)})
            print(f"[analyze_groups] {group.name}: {exc}", file=sys.stderr)
    summary = {
        "schema": "linkerhand_group_analysis_v1",
        "session": str(args.session),
        "groups": results,
        "ready_groups": sum(bool(item.get("ready")) for item in results),
    }
    (args.session / "analysis_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
