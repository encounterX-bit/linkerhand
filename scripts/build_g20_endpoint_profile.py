#!/usr/bin/env python3
"""Build stable G20 task-end joint templates from an ACT source manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


JOINT_COUNT = 20
RESERVED_IDX = (11, 12, 13, 14)
# Task completion is defined by the thumb channels and the five finger bases.
# Lateral spread and non-thumb tip feedback vary substantially with cube load
# and MediaPipe residuals even when the visible flip has already completed.
DEFAULT_ENDPOINT_INDICES = (0, 1, 2, 3, 4, 5, 10, 15)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--artifact-root", type=Path, required=True)
    ap.add_argument("--tail-frames", type=int, default=30)
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()
    args.artifact_root = args.artifact_root.expanduser().resolve()
    if args.output is None:
        args.output = args.artifact_root / "dataset" / "g20_endpoint_profiles.json"
    else:
        args.output = args.output.expanduser().resolve()
    if args.tail_frames <= 0:
        ap.error("--tail-frames must be positive")
    return args


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"invalid JSON at {path}:{line_number}") from exc
            position = row.get("joint_pos")
            if not isinstance(position, list) or len(position) < JOINT_COUNT:
                raise RuntimeError(f"missing 20-D joint_pos at {path}:{line_number}")
            rows.append(row)
    if not rows:
        raise RuntimeError(f"empty episode: {path}")
    return rows


def main() -> int:
    args = parse_args()
    manifest_path = args.artifact_root / "dataset" / "g20_source_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    roots = {Path(value).name: Path(value) for value in manifest["data_roots"]}
    templates: list[dict[str, Any]] = []
    for episode in manifest["episodes"]:
        source = str(episode["source"])
        root_name, separator, relative = source.partition("/")
        if not separator or root_name not in roots:
            raise RuntimeError(f"cannot resolve source episode {source!r}")
        samples_path = roots[root_name] / relative / "samples.jsonl"
        rows = load_rows(samples_path)
        tail = rows[-min(args.tail_frames, len(rows)) :]
        positions = np.asarray(
            [row["joint_pos"][:JOINT_COUNT] for row in tail], dtype=np.float32
        )
        median = np.median(positions, axis=0)
        templates.append(
            {
                "source": source,
                "dataset_episode_index": int(episode["dataset_episode_index"]),
                "position": np.rint(median).astype(np.int32).tolist(),
                "tail_frames": len(tail),
                "tail_max_range": float(np.ptp(positions, axis=0).max()),
            }
        )
    profile = {
        "schema": "g20_demo_endpoint_profile_v1",
        "artifact_root": str(args.artifact_root),
        "source_manifest": str(manifest_path),
        "active_indices": list(DEFAULT_ENDPOINT_INDICES),
        "tail_frames": args.tail_frames,
        "templates": templates,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(profile, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"[endpoint_profile] wrote {len(templates)} templates to {args.output}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
