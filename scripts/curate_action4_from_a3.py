#!/usr/bin/env python3
"""Derive completed hybrid A4 episodes beginning at the A3 endpoint.

The raw recorder sessions are immutable.  This script finds the first stable
measured A3 endpoint, verifies the actual A4 thumb order (left-align, close,
then turn).  The stable A4 endpoint is used only to validate completion: the
derived fragment ends immediately before that stable endpoint so it cannot
teach ACT that A4 is a terminal/hold state.  The original full-task demos keep
exclusive supervision of the A4 -> A1/A5 continuation.  Four-finger joints are
deliberately not compared with A4 because they were controlled by MediaPipe
after the operator pressed T.

The default mode is a read-only audit.  Pass ``--write`` to create JSONL files
whose image paths still point at the original JPEGs.
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
JOINT_COUNT = 20
RESERVED_IDX = (11, 12, 13, 14)
ACTIVE_IDX = tuple(i for i in range(JOINT_COUNT) if i not in RESERVED_IDX)
THUMB_IDX = (0, 5, 10, 15)
THUMB_ORIENTATION_IDX = (0, 5, 10)


@dataclass(frozen=True)
class StableRun:
    start: int
    end: int

    @property
    def frames(self) -> int:
        return self.end - self.start + 1


@dataclass(frozen=True)
class CuratedEpisode:
    source: Path
    rows: tuple[dict[str, Any], ...]
    start_frame: int
    end_frame: int
    left_frame: int
    close_frame: int
    endpoint_frame: int


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--library",
        type=Path,
        default=Path("data/action_library/g20_right/core_actions_v1"),
    )
    parser.add_argument(
        "--source-session",
        type=Path,
        action="append",
        required=True,
        help="raw recorder session; repeat for multiple sessions",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            "data/act_demos_clean/action4_hybrid_transition_from_a3_v1"
        ),
    )
    parser.add_argument("--start-tolerance", type=float, default=10.0)
    parser.add_argument("--left-tolerance", type=float, default=10.0)
    parser.add_argument("--close-tolerance", type=float, default=10.0)
    parser.add_argument("--endpoint-tolerance", type=float, default=10.0)
    parser.add_argument("--start-confirm-frames", type=int, default=5)
    parser.add_argument("--stage-confirm-frames", type=int, default=3)
    parser.add_argument("--endpoint-confirm-frames", type=int, default=15)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    args.library = args.library.expanduser().resolve()
    args.source_session = [path.expanduser().resolve() for path in args.source_session]
    args.output_root = args.output_root.expanduser().resolve()
    if not args.library.is_dir():
        parser.error(f"--library does not exist: {args.library}")
    missing = [path for path in args.source_session if not path.is_dir()]
    if missing:
        parser.error("missing --source-session: " + ", ".join(map(str, missing)))
    tolerances = (
        args.start_tolerance,
        args.left_tolerance,
        args.close_tolerance,
        args.endpoint_tolerance,
    )
    confirms = (
        args.start_confirm_frames,
        args.stage_confirm_frames,
        args.endpoint_confirm_frames,
    )
    if any(not np.isfinite(value) or value < 0 for value in tolerances):
        parser.error("tolerances must be finite and nonnegative")
    if any(value <= 0 for value in confirms):
        parser.error("confirm-frame counts must be positive")
    if args.overwrite and not args.write:
        parser.error("--overwrite requires --write")
    return args


def stable_runs(mask: Sequence[bool], minimum_frames: int) -> list[StableRun]:
    values = np.asarray(mask, dtype=bool).reshape(-1)
    runs: list[StableRun] = []
    begin: int | None = None
    for index, active in enumerate(values):
        if active and begin is None:
            begin = index
        if begin is not None and (not active or index == len(values) - 1):
            end = index if active and index == len(values) - 1 else index - 1
            if end - begin + 1 >= minimum_frames:
                runs.append(StableRun(begin, end))
            begin = None
    return runs


def first_run_after(
    mask: Sequence[bool], minimum_frames: int, after: int
) -> StableRun | None:
    for run in stable_runs(mask, minimum_frames):
        if run.end >= after:
            clipped = StableRun(max(run.start, after), run.end)
            if clipped.frames >= minimum_frames:
                return clipped
    return None


def read_rows(episode: Path) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    with (episode / "samples.jsonl").open("r", encoding="utf-8") as stream:
        for line in stream:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            image = row.get("image_path")
            state = row.get("joint_pos")
            action = row.get("last_action")
            if not (
                isinstance(image, str)
                and (episode / image).is_file()
                and isinstance(state, list)
                and len(state) >= JOINT_COUNT
                and isinstance(action, list)
                and len(action) >= JOINT_COUNT
            ):
                continue
            rows.append(row)
    return tuple(rows)


def curate_episode(
    episode: Path,
    rows: tuple[dict[str, Any], ...],
    *,
    a3_endpoint: np.ndarray,
    a4_left: np.ndarray,
    a4_endpoint: np.ndarray,
    start_tolerance: float,
    left_tolerance: float,
    close_tolerance: float,
    endpoint_tolerance: float,
    start_confirm_frames: int,
    stage_confirm_frames: int,
    endpoint_confirm_frames: int,
) -> tuple[CuratedEpisode | None, str]:
    if not rows:
        return None, "no_usable_rows"
    state = np.asarray([row["joint_pos"][:JOINT_COUNT] for row in rows], dtype=np.float32)
    start_error = np.max(
        np.abs(state[:, list(ACTIVE_IDX)] - a3_endpoint[list(ACTIVE_IDX)]), axis=1
    )
    start = first_run_after(
        start_error <= start_tolerance, start_confirm_frames, 0
    )
    if start is None:
        return None, "no_stable_a3_endpoint"

    left_error = np.max(
        np.abs(
            state[:, list(THUMB_ORIENTATION_IDX)]
            - a4_left[list(THUMB_ORIENTATION_IDX)]
        ),
        axis=1,
    )
    left = first_run_after(
        left_error <= left_tolerance, stage_confirm_frames, start.start
    )
    if left is None:
        return None, "no_left_alignment"

    close_error = np.abs(state[:, 15] - a4_left[15])
    close = first_run_after(
        close_error <= close_tolerance, stage_confirm_frames, left.start
    )
    if close is None:
        return None, "no_thumb_tip_close"

    endpoint_error = np.max(
        np.abs(state[:, list(THUMB_IDX)] - a4_endpoint[list(THUMB_IDX)]), axis=1
    )
    endpoint = first_run_after(
        endpoint_error <= endpoint_tolerance,
        endpoint_confirm_frames,
        close.start,
    )
    if endpoint is None:
        return None, "no_stable_a4_endpoint"
    # Do not include observations at the stable endpoint.  Otherwise these
    # partial A4 recordings would add terminal/hold supervision exactly where
    # the full-task policy must choose its A1/A5 continuation.
    end_frame = endpoint.start - 1
    if end_frame <= start.start:
        return None, "no_nonterminal_a4_transition"
    return (
        CuratedEpisode(
            episode,
            rows,
            start.start,
            end_frame,
            left.start,
            close.start,
            endpoint.start,
        ),
        "kept",
    )


def write_episode(
    output_root: Path, episode_index: int, curated: CuratedEpisode
) -> None:
    destination = output_root / f"episode_{episode_index:03d}"
    destination.mkdir(parents=True, exist_ok=False)
    selected = curated.rows[curated.start_frame : curated.end_frame + 1]
    first_elapsed = float(selected[0].get("elapsed", 0.0) or 0.0)
    with (destination / "samples.jsonl").open("w", encoding="utf-8") as stream:
        for new_index, source_row in enumerate(selected):
            row = dict(source_row)
            source_image = (
                curated.source / str(source_row["image_path"])
            ).resolve()
            row.update(
                {
                    "index": new_index,
                    "episode": episode_index,
                    "elapsed": max(
                        0.0,
                        float(source_row.get("elapsed", first_elapsed) or first_elapsed)
                        - first_elapsed,
                    ),
                    "image_path": str(source_image),
                    "source_episode": str(curated.source),
                    "source_frame_index": curated.start_frame + new_index,
                    "derived_segment": "a3_endpoint_to_a4_endpoint",
                }
            )
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    metadata = {
        "source_episode": str(curated.source),
        "source_start_frame": curated.start_frame,
        "source_end_frame": curated.end_frame,
        "left_frame": curated.left_frame,
        "close_frame": curated.close_frame,
        "endpoint_frame": curated.endpoint_frame,
        "frames": curated.end_frame - curated.start_frame + 1,
        "hybrid_control": "A4 thumb plus MediaPipe four fingers",
    }
    (destination / "episode.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = json.loads((args.library / "manifest.json").read_text(encoding="utf-8"))
    by_id = {int(item["id"]): item for item in manifest["primitives"]}
    a3 = np.load(args.library / by_id[3]["robot_trajectory"]).astype(np.float32)
    a4 = np.load(args.library / by_id[4]["robot_trajectory"]).astype(np.float32)
    a3_endpoint = a3[-1]
    a4_left = a4[0]
    a4_endpoint = a4[-1]

    kept: list[CuratedEpisode] = []
    report_rows: list[dict[str, Any]] = []
    for session in args.source_session:
        for samples in sorted(session.glob("episode_*/samples.jsonl")):
            episode = samples.parent.resolve()
            rows = read_rows(episode)
            curated, reason = curate_episode(
                episode,
                rows,
                a3_endpoint=a3_endpoint,
                a4_left=a4_left,
                a4_endpoint=a4_endpoint,
                start_tolerance=args.start_tolerance,
                left_tolerance=args.left_tolerance,
                close_tolerance=args.close_tolerance,
                endpoint_tolerance=args.endpoint_tolerance,
                start_confirm_frames=args.start_confirm_frames,
                stage_confirm_frames=args.stage_confirm_frames,
                endpoint_confirm_frames=args.endpoint_confirm_frames,
            )
            record: dict[str, Any] = {
                "source_episode": str(episode),
                "source_frames": len(rows),
                "reason": reason,
            }
            if curated is not None:
                kept.append(curated)
                record.update(
                    {
                        "start_frame": curated.start_frame,
                        "end_frame": curated.end_frame,
                        "frames": curated.end_frame - curated.start_frame + 1,
                        "left_frame": curated.left_frame,
                        "close_frame": curated.close_frame,
                        "endpoint_frame": curated.endpoint_frame,
                    }
                )
                print(
                    f"[keep] {session.name}/{episode.name} "
                    f"frames={record['frames']} stages="
                    f"{curated.start_frame}->{curated.left_frame}->"
                    f"{curated.close_frame}->{curated.endpoint_frame}",
                    flush=True,
                )
            else:
                print(f"[drop] {session.name}/{episode.name}: {reason}", flush=True)
            report_rows.append(record)

    report = {
        "schema": "linkerhand_action4_transition_from_a3_clean_v1",
        "library": str(args.library),
        "source_sessions": [str(path) for path in args.source_session],
        "output_root": str(args.output_root),
        "kept_episodes": len(kept),
        "kept_frames": sum(item.end_frame - item.start_frame + 1 for item in kept),
        "episodes": report_rows,
    }
    print(
        f"[summary] kept={report['kept_episodes']} frames={report['kept_frames']} "
        f"write={args.write}",
        flush=True,
    )
    if not kept:
        raise RuntimeError("no completed A4 episodes passed curation")
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
        "task_id": (
            "action_library_hybrid_demo_v1_a4_transition_from_a3_clean_v1"
        ),
        "derived": True,
        "source_sessions": [str(path) for path in args.source_session],
        "segment": (
            "A3 endpoint through hybrid A4 approach; stable A4 endpoint "
            "validated but excluded so A1/A5 continuation remains nonterminal"
        ),
    }
    (args.output_root / "session.json").write_text(
        json.dumps(session, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    for index, curated in enumerate(kept):
        write_episode(args.output_root, index, curated)
    (args.output_root / "curation_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"[write] derived A4 dataset: {args.output_root}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        print(f"error: {exc}")
        raise SystemExit(2)
