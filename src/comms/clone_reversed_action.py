#!/usr/bin/env python3
"""Add a time-reversed copy of one primitive to an action library."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np

from src.comms.action_library import ActionLibrary


def clone_reversed_action(
    library: Path,
    *,
    source_id: int,
    primitive_id: int,
    name: str,
) -> dict[str, Any]:
    """Create an independent reverse primitive while preserving its source."""
    library = Path(library).resolve()
    manifest_path = library / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = manifest.get("primitives", [])
    source = next(
        (record for record in records if int(record["id"]) == int(source_id)),
        None,
    )
    if source is None:
        raise ValueError(f"source primitive {source_id} does not exist")
    if any(int(record["id"]) == int(primitive_id) for record in records):
        raise ValueError(f"destination primitive {primitive_id} already exists")
    if primitive_id <= 0 or source_id == primitive_id:
        raise ValueError("destination ID must be positive and differ from source ID")
    if not name or any(character.isspace() for character in name):
        raise ValueError("name must be non-empty and contain no whitespace")

    source_trajectory = np.load(
        library / source["robot_trajectory"], allow_pickle=False
    )
    source_templates = [
        np.load(library / relative, allow_pickle=False)
        for relative in source.get("human_templates", [])
    ]
    if source_trajectory.ndim != 2 or source_trajectory.shape[1] != 20:
        raise ValueError("source robot trajectory must have shape (T,20)")
    if not source_templates:
        raise ValueError("source primitive has no human templates")

    folder_name = f"primitive_{primitive_id:03d}_{name}"
    destination = library / folder_name
    if destination.exists():
        raise ValueError(f"destination folder already exists: {destination}")
    destination.mkdir(parents=False)

    original_manifest = manifest_path.read_bytes()
    try:
        trajectory_path = destination / "robot_trajectory.npy"
        np.save(
            trajectory_path,
            source_trajectory[::-1].copy(),
            allow_pickle=False,
        )
        template_paths: list[str] = []
        for index, template in enumerate(source_templates):
            path = destination / f"human_take_{index:03d}.npy"
            np.save(path, template[::-1].copy(), allow_pickle=False)
            template_paths.append(str(path.relative_to(library)))

        record = {
            "id": int(primitive_id),
            "name": name,
            "robot_trajectory": str(trajectory_path.relative_to(library)),
            "human_templates": template_paths,
            "threshold": float(source.get("threshold", 0.18)),
            "interruptible": bool(source.get("interruptible", False)),
            "cooldown_frames": max(0, int(source.get("cooldown_frames", 10))),
            "manual_from_start": True,
            "derived_from_primitive_id": int(source_id),
            "derived_operation": "reverse_time",
        }
        records.append(record)
        records.sort(key=lambda item: int(item["id"]))
        temporary_manifest = manifest_path.with_suffix(".json.tmp")
        temporary_manifest.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        temporary_manifest.replace(manifest_path)
        ActionLibrary.load(library)
    except Exception:
        manifest_path.write_bytes(original_manifest)
        shutil.rmtree(destination)
        raise

    return {
        "library": library,
        "source_id": int(source_id),
        "primitive_id": int(primitive_id),
        "trajectory_frames": int(len(source_trajectory)),
        "templates": int(len(source_templates)),
        "first_pose": source_trajectory[-1].astype(float).tolist(),
        "last_pose": source_trajectory[0].astype(float).tolist(),
    }


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--source-id", type=int, required=True)
    parser.add_argument("--primitive-id", type=int, required=True)
    parser.add_argument("--name", required=True)
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    try:
        result = clone_reversed_action(
            args.library,
            source_id=args.source_id,
            primitive_id=args.primitive_id,
            name=args.name,
        )
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"[clone_reverse] {exc}", file=sys.stderr)
        return 2
    print(
        f"[clone_reverse] added {result['primitive_id']} as reverse of "
        f"{result['source_id']}; frames={result['trajectory_frames']} "
        f"templates={result['templates']} library={result['library']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
