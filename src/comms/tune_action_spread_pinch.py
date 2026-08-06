#!/usr/bin/env python3
"""Prepend a gradual four-finger spread pinch to one library primitive."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

from src.comms.action_library import ActionLibrary, RESERVED_IDX


SPREAD_IDX = (6, 7, 8, 9)
OUTER_FLEXION_IDX = (1, 4, 16, 19)


def _parse_spread_target(value: str) -> tuple[float, float, float, float]:
    try:
        values = tuple(float(item.strip()) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "spread target must contain four comma-separated numbers"
        ) from exc
    if (
        len(values) != 4
        or not all(np.isfinite(values))
        or any(item < 0.0 or item > 255.0 for item in values)
    ):
        raise argparse.ArgumentTypeError(
            "spread target must contain four finite SDK values in 0..255"
        )
    return values


def tune_action_spread_pinch(
    library: Path,
    *,
    primitive_id: int,
    spread_target: Sequence[float],
    transition_frames: int,
    outer_flexion_max: Optional[float] = None,
    archive_name: Optional[str] = None,
) -> dict[str, Any]:
    """Hold the first pose while pinching q6..q9, then run the original action."""
    library = Path(library).resolve()
    manifest_path = library / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    record = next(
        (
            item
            for item in manifest.get("primitives", [])
            if int(item["id"]) == int(primitive_id)
        ),
        None,
    )
    if record is None:
        raise ValueError(f"primitive {primitive_id} does not exist")
    target = np.asarray(spread_target, dtype=np.float32).reshape(-1)
    if (
        target.shape != (4,)
        or not np.all(np.isfinite(target))
        or np.any(target < 0.0)
        or np.any(target > 255.0)
    ):
        raise ValueError("spread target must contain four SDK values in 0..255")
    if int(transition_frames) <= 0:
        raise ValueError("transition frames must be positive")
    if outer_flexion_max is not None and (
        not np.isfinite(outer_flexion_max)
        or outer_flexion_max < 0.0
        or outer_flexion_max > 255.0
    ):
        raise ValueError("outer flexion maximum must be in SDK range 0..255")

    trajectory_path = library / record["robot_trajectory"]
    trajectory = np.load(trajectory_path, allow_pickle=False)
    if trajectory.ndim != 2 or trajectory.shape[1] != 20 or not len(trajectory):
        raise ValueError("robot trajectory must have shape (T,20)")

    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    label = archive_name or (
        f"{timestamp}_before_action_{int(primitive_id):03d}_spread_pinch"
    )
    archive = library / "archive" / label
    if archive.exists():
        raise ValueError(f"archive already exists: {archive}")
    archive.mkdir(parents=True)
    shutil.copy2(manifest_path, archive / "manifest.json")
    shutil.copy2(trajectory_path, archive / "robot_trajectory.npy")

    existing_override = record.get("spread_pinch_override")
    base_original_frames = int(len(trajectory))
    previous_target: Optional[list[float]] = None
    if isinstance(existing_override, dict):
        existing_transition_frames = int(
            existing_override.get("transition_frames", -1)
        )
        base_original_frames = int(
            existing_override.get("original_frames", -1)
        )
        expected_frames = base_original_frames + existing_transition_frames
        if (
            existing_transition_frames <= 0
            or base_original_frames <= 0
            or expected_frames != len(trajectory)
        ):
            raise ValueError(
                "existing spread_pinch_override does not match trajectory"
            )
        previous_target = [
            float(item)
            for item in existing_override.get(
                "target_q6_q7_q8_q9", []
            )
        ]
        if len(previous_target) != 4:
            raise ValueError("existing spread pinch target is invalid")
        # Drop the installed prelude and retain its underlying action tail.
        tail = trajectory[
            existing_transition_frames + 1:
        ].astype(np.float32, copy=True)
    else:
        tail = trajectory[1:].astype(np.float32, copy=True)

    start = trajectory[0].astype(np.float32, copy=True)
    pinched = start.copy()
    pinched[list(SPREAD_IDX)] = target
    prelude = np.linspace(
        start,
        pinched,
        int(transition_frames) + 1,
        dtype=np.float32,
    )
    if len(tail):
        tail[:, list(SPREAD_IDX)] = target
        if outer_flexion_max is not None:
            tail[:, list(OUTER_FLEXION_IDX)] = np.minimum(
                tail[:, list(OUTER_FLEXION_IDX)],
                float(outer_flexion_max),
            )
    tuned = np.concatenate((prelude, tail), axis=0)
    tuned[:, list(RESERVED_IDX)] = 255.0
    tuned = np.clip(tuned, 0.0, 255.0)

    record["spread_pinch_override"] = {
        "target_q6_q7_q8_q9": target.astype(float).tolist(),
        "transition_frames": int(transition_frames),
        "original_frames": base_original_frames,
        "new_frames": int(len(tuned)),
        "ordering": (
            "reach original first pose, pinch while holding it, then play "
            "the original action with the pinch held"
        ),
        "target_source": (
            "operator-tuned from official G20 GUI preset 动作2"
            if existing_override is not None
            else "official G20 GUI preset 动作2"
        ),
        "archive": str(archive.relative_to(library)),
    }
    if previous_target is not None:
        record["spread_pinch_override"]["previous_target_q6_q7_q8_q9"] = (
            previous_target
        )
    if outer_flexion_max is not None:
        record["outer_finger_hold_override"] = {
            "indices": list(OUTER_FLEXION_IDX),
            "max_sdk_value": float(outer_flexion_max),
            "meaning": (
                "index and little base/tip stay slightly flexed while "
                "middle and ring complete the reverse/open motion"
            ),
            "archive": str(archive.relative_to(library)),
        }
    record["best_effort_spread_feedback"] = True
    temporary_trajectory = trajectory_path.with_name(
        trajectory_path.stem + ".tmp.npy"
    )
    temporary_manifest = manifest_path.with_suffix(".json.tmp")
    try:
        np.save(temporary_trajectory, tuned, allow_pickle=False)
        temporary_manifest.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        temporary_trajectory.replace(trajectory_path)
        temporary_manifest.replace(manifest_path)
        ActionLibrary.load(library)
    except Exception:
        shutil.copy2(archive / "robot_trajectory.npy", trajectory_path)
        shutil.copy2(archive / "manifest.json", manifest_path)
        temporary_trajectory.unlink(missing_ok=True)
        temporary_manifest.unlink(missing_ok=True)
        raise

    return {
        "primitive_id": int(primitive_id),
        "original_frames": int(len(trajectory)),
        "new_frames": int(len(tuned)),
        "start_spread": trajectory[0, list(SPREAD_IDX)].astype(float).tolist(),
        "target_spread": target.astype(float).tolist(),
        "archive": archive,
    }


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--primitive-id", type=int, required=True)
    parser.add_argument(
        "--spread-target",
        type=_parse_spread_target,
        required=True,
        metavar="Q6,Q7,Q8,Q9",
    )
    parser.add_argument("--transition-frames", type=int, default=24)
    parser.add_argument(
        "--outer-flexion-max",
        type=float,
        default=None,
        help=(
            "optional maximum SDK value for q1/q4/q16/q19 after the "
            "pinch prelude; lower than 255 keeps index/little flexed"
        ),
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    try:
        result = tune_action_spread_pinch(
            args.library,
            primitive_id=args.primitive_id,
            spread_target=args.spread_target,
            transition_frames=args.transition_frames,
            outer_flexion_max=args.outer_flexion_max,
        )
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"[spread_pinch] {exc}", file=sys.stderr)
        return 2
    print(
        f"[spread_pinch] action={result['primitive_id']} "
        f"frames={result['original_frames']}->{result['new_frames']} "
        f"spread={result['start_spread']}->{result['target_spread']} "
        f"archive={result['archive']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
