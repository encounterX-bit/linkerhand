#!/usr/bin/env python3
"""Audit local LinkerHand data for a Being-H0.8-style tactile baseline.

This is read-only. It scans ``samples.jsonl`` files and reports synchronized
vision, state, action, coarse tactile, and full 12x6 fingertip-matrix coverage.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    return parser.parse_args()


def valid_vector(value: Any, length: int) -> bool:
    return isinstance(value, list) and len(value) >= length


def audit(data_root: Path) -> dict[str, Any]:
    root = data_root.expanduser().resolve()
    counts: Counter[str] = Counter()
    file_counts: Counter[str] = Counter()
    jsonl_files = sorted(root.rglob("samples.jsonl"))

    for samples_path in jsonl_files:
        local: Counter[str] = Counter()
        matrix_path = samples_path.parent / "matrices.npz"
        with samples_path.open(encoding="utf-8") as stream:
            for line in stream:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    counts["invalid_json"] += 1
                    continue
                counts["frames"] += 1
                local["frames"] += 1
                if valid_vector(row.get("joint_pos"), 20):
                    counts["joint20"] += 1
                if valid_vector(row.get("last_action"), 20):
                    counts["action20"] += 1
                image_path = row.get("image_path")
                if image_path and (samples_path.parent / str(image_path)).is_file():
                    counts["image"] += 1
                if valid_vector(row.get("mass_values"), 6):
                    counts["mass6"] += 1
                    local["mass6"] += 1
                if valid_vector(row.get("contact_6"), 6):
                    counts["contact6"] += 1
                    local["contact6"] += 1
                if row.get("has_matrix") and matrix_path.is_file():
                    counts["matrix12x6"] += 1
                    local["matrix12x6"] += 1
                if (
                    valid_vector(row.get("joint_pos"), 20)
                    and valid_vector(row.get("last_action"), 20)
                    and image_path
                    and (samples_path.parent / str(image_path)).is_file()
                    and valid_vector(row.get("mass_values"), 6)
                    and valid_vector(row.get("contact_6"), 6)
                ):
                    counts["h08_lite_usable"] += 1
        if local["mass6"] and local["contact6"]:
            file_counts["coarse_tactile_episodes"] += 1
        if local["matrix12x6"]:
            file_counts["matrix_tactile_episodes"] += 1

    frames = counts["frames"]
    coverage = {
        key: (counts[key] / frames if frames else 0.0)
        for key in (
            "joint20",
            "action20",
            "image",
            "mass6",
            "contact6",
            "matrix12x6",
            "h08_lite_usable",
        )
    }
    return {
        "schema": "linkerhand_h08_lite_audit_v1",
        "data_root": str(root),
        "jsonl_files": len(jsonl_files),
        "counts": dict(counts),
        "coverage": coverage,
        "episode_files": dict(file_counts),
    }


def main() -> int:
    args = parse_args()
    report = audit(args.data_root)
    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0

    counts = report["counts"]
    print(f"files={report['jsonl_files']} frames={counts.get('frames', 0)}")
    for key, ratio in report["coverage"].items():
        print(f"{key:18s} {counts.get(key, 0):8d}  {ratio:6.1%}")
    for key, value in report["episode_files"].items():
        print(f"{key:24s} {value:8d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
