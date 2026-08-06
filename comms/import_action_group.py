#!/usr/bin/env python3
"""Import one audited action group into a MediaPipe G20 action library."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np

from src.comms.action_library import (
    ACTIVE_IDX,
    ActionLibrary,
    FEATURE_PROFILE_FULL,
    FEATURE_PROFILES,
    dtw_distance,
    landmark_feature,
)
from src.comms.replay_action_group import densify_trajectory, load_replay_group


G20_FOUR_FINGER_SPREAD_IDX = (6, 7, 8, 9)
G20_THUMB_IDX = (0, 5, 10, 15)


def _load_human_takes(group: Path) -> list[np.ndarray]:
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
            if landmarks.shape != (21, 3) or not np.all(np.isfinite(landmarks)):
                raise ValueError("human take contains invalid hand-base landmarks")
            frames.append(landmarks)
        if len(frames) < 6:
            raise ValueError(
                f"take {item.get('take_index')} has only {len(frames)} fresh frames"
            )
        takes.append(np.stack(frames))
    if len(takes) < 2:
        raise ValueError("an action-library primitive needs at least two human takes")
    return takes


def _automatic_threshold(
    takes: list[np.ndarray],
    *,
    feature_profile: str = FEATURE_PROFILE_FULL,
) -> float:
    features = [
        np.stack([
            landmark_feature(frame, feature_profile=feature_profile)
            for frame in take
        ])
        for take in takes
    ]
    # A loose "largest pairwise distance" threshold is dominated by two
    # outlier repetitions and can overlap a neighbouring gesture.  What online
    # nearest-template matching needs is coverage by at least one same-class
    # template, so use the worst leave-one-out nearest neighbour plus a small
    # capture/noise allowance.
    nearest = [
        min(
            dtw_distance(query, candidate)
            for other_index, candidate in enumerate(features)
            if query_index != other_index
        )
        for query_index, query in enumerate(features)
    ]
    return float(np.clip(max(nearest) + 0.025, 0.06, 0.18))


def _waypoint_diagnostics(group: Path) -> dict[str, Any]:
    payload = json.loads(
        (group / "robot" / "waypoints.json").read_text(encoding="utf-8")
    )
    waypoints = payload.get("waypoints", [])
    if not waypoints:
        raise ValueError("group has no recorded command/state waypoints")
    commands = np.asarray([item["command"] for item in waypoints], dtype=np.float32)
    states = np.asarray([item["state"] for item in waypoints], dtype=np.float32)
    if commands.shape != states.shape or commands.ndim != 2 or commands.shape[1] != 20:
        raise ValueError("recorded command/state waypoints must both have shape (N,20)")
    errors = np.abs(commands - states)
    primary_idx = tuple(i for i in ACTIVE_IDX if i not in G20_FOUR_FINGER_SPREAD_IDX)
    return {
        "max_primary_error": float(np.max(errors[:, list(primary_idx)])),
        "max_spread_error": float(
            np.max(errors[:, list(G20_FOUR_FINGER_SPREAD_IDX)])
        ),
        "thumb_command_span": np.ptp(
            commands[:, list(G20_THUMB_IDX)], axis=0
        ).astype(float).tolist(),
    }


def import_group(args: argparse.Namespace) -> dict[str, Any]:
    unified_group = getattr(args, "group", None)
    human_group_arg = getattr(args, "human_group", None) or unified_group
    robot_group_arg = getattr(args, "robot_group", None) or unified_group
    if human_group_arg is None or robot_group_arg is None:
        raise ValueError(
            "provide --group, or provide both --human-group and --robot-group"
        )
    human_group = Path(human_group_arg).resolve()
    robot_group = Path(robot_group_arg).resolve()
    allow_spread_coupling = bool(
        getattr(args, "allow_spread_coupling_error", False)
    )
    max_spread_error = float(getattr(args, "max_recorded_spread_error", 45.0))
    source = load_replay_group(
        robot_group,
        rate=args.fps,
        max_recorded_state_error=(
            max_spread_error
            if allow_spread_coupling
            else args.max_recorded_state_error
        ),
    )
    if source.issues:
        raise ValueError("group replay preflight blocked: " + "; ".join(source.issues))
    diagnostics = _waypoint_diagnostics(source.path)
    if diagnostics["max_primary_error"] > float(args.max_recorded_state_error):
        raise ValueError(
            "recorded primary-joint command/state error "
            f"{diagnostics['max_primary_error']:.0f} exceeds limit "
            f"{float(args.max_recorded_state_error):.0f}"
        )
    if allow_spread_coupling and diagnostics["max_spread_error"] > max_spread_error:
        raise ValueError(
            "recorded spread coupling error "
            f"{diagnostics['max_spread_error']:.0f} exceeds limit {max_spread_error:.0f}"
        )
    max_thumb_span = float(getattr(args, "max_thumb_command_span", 2.0))
    if bool(getattr(args, "require_static_thumb", False)) and max(
        diagnostics["thumb_command_span"]
    ) > max_thumb_span:
        raise ValueError(
            f"thumb command span {diagnostics['thumb_command_span']} exceeds "
            f"static-thumb limit {max_thumb_span:.0f}"
        )
    library = args.library.resolve()
    manifest_path = library / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema") != ActionLibrary.SCHEMA:
            raise ValueError(f"unsupported action library schema {manifest.get('schema')!r}")
    else:
        requested_profile = getattr(args, "feature_profile", None)
        manifest = {
            "schema": ActionLibrary.SCHEMA,
            "hand_model": "g20_palm_touch",
            "joint_space": "sdk_range_0_255",
            "fps": float(args.fps),
            "feature_profile": requested_profile or FEATURE_PROFILE_FULL,
            "primitives": [],
        }
    feature_profile = str(manifest.get("feature_profile", FEATURE_PROFILE_FULL))
    requested_profile = getattr(args, "feature_profile", None)
    if feature_profile not in FEATURE_PROFILES:
        raise ValueError(f"unsupported landmark feature profile {feature_profile!r}")
    if requested_profile is not None and requested_profile != feature_profile:
        raise ValueError(
            f"library feature profile is {feature_profile!r}, not {requested_profile!r}"
        )
    manifest["feature_profile"] = feature_profile
    takes = _load_human_takes(human_group)
    trajectory = densify_trajectory(source.trajectory, max_step=args.trajectory_max_step)
    threshold = (
        _automatic_threshold(takes, feature_profile=feature_profile)
        if args.threshold is None
        else float(args.threshold)
    )
    records = manifest.setdefault("primitives", [])
    existing = next(
        (item for item in records if int(item["id"]) == args.primitive_id), None
    )
    if existing is not None and not args.replace:
        raise ValueError(
            f"primitive id {args.primitive_id} already exists; use --replace explicitly"
        )

    folder_name = f"primitive_{args.primitive_id:03d}_{args.name}"
    folder = library / folder_name
    folder.mkdir(parents=True, exist_ok=True)
    np.save(folder / "robot_trajectory.npy", trajectory, allow_pickle=False)
    template_paths = []
    for index, take in enumerate(takes):
        path = folder / f"human_take_{index:03d}.npy"
        np.save(path, take, allow_pickle=False)
        template_paths.append(str(path.relative_to(library)))
    record = {
        "id": int(args.primitive_id),
        "name": str(args.name),
        "robot_trajectory": str(
            (folder / "robot_trajectory.npy").relative_to(library)
        ),
        "human_templates": template_paths,
        "threshold": threshold,
        "interruptible": False,
        "cooldown_frames": max(0, int(args.cooldown_frames)),
        "source_group": str(source.path),
        "human_source_group": str(human_group),
        "robot_source_group": str(source.path),
        "source_max_recorded_state_error": source.max_recorded_error,
        "source_max_recorded_primary_error": diagnostics["max_primary_error"],
        "source_max_recorded_spread_error": diagnostics["max_spread_error"],
        "source_thumb_command_span": diagnostics["thumb_command_span"],
    }
    if existing is None:
        records.append(record)
    else:
        records[records.index(existing)] = record
    records.sort(key=lambda item: int(item["id"]))
    library.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    ActionLibrary.load(library)
    return {
        "library": library,
        "templates": len(takes),
        "trajectory_frames": len(trajectory),
        "threshold": threshold,
    }


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--group",
        type=Path,
        help="one group containing both human takes and robot waypoints",
    )
    parser.add_argument(
        "--human-group",
        type=Path,
        help="optional separate group supplying only the human takes",
    )
    parser.add_argument(
        "--robot-group",
        type=Path,
        help="optional separate group supplying only the robot waypoints",
    )
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--primitive-id", type=int, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--trajectory-max-step", type=int, default=5)
    parser.add_argument("--max-recorded-state-error", type=float, default=10.0)
    parser.add_argument(
        "--allow-spread-coupling-error",
        action="store_true",
        help="audit q6..q9 against a separate coupling limit",
    )
    parser.add_argument("--max-recorded-spread-error", type=float, default=45.0)
    parser.add_argument("--require-static-thumb", action="store_true")
    parser.add_argument("--max-thumb-command-span", type=float, default=2.0)
    parser.add_argument("--threshold", type=float)
    parser.add_argument(
        "--feature-profile",
        choices=FEATURE_PROFILES,
        default=None,
        help="feature profile for a new library; existing libraries keep their manifest profile",
    )
    parser.add_argument("--cooldown-frames", type=int, default=20)
    parser.add_argument("--replace", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    if args.fps <= 0 or args.trajectory_max_step <= 0:
        print("[import_group] fps and trajectory max step must be positive", file=sys.stderr)
        return 2
    if args.threshold is not None and args.threshold <= 0:
        print("[import_group] threshold must be positive", file=sys.stderr)
        return 2
    if args.max_recorded_spread_error <= 0 or args.max_thumb_command_span < 0:
        print(
            "[import_group] spread error must be positive and thumb span nonnegative",
            file=sys.stderr,
        )
        return 2
    try:
        result = import_group(args)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"[import_group] {exc}", file=sys.stderr)
        return 2
    print(
        f"[import_group] saved {result['library']} templates={result['templates']} "
        f"trajectory_frames={result['trajectory_frames']} "
        f"threshold={result['threshold']:.4f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
