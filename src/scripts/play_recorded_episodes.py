#!/usr/bin/env python3
"""Play recorded camera episodes in order, advancing with SPACE.

This utility is visualization-only: it does not import ROS and never publishes
commands to a hand. Each episode is played once at the requested frame rate,
then the last frame remains visible until SPACE advances to the next episode.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2


DEFAULT_DATA_ROOT = Path("data/20260715_112309_rotate_object_90deg_cw")
WINDOW_NAME = "Recorded G20 episodes"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument(
        "--start-episode",
        default=None,
        help="episode directory name to start from, for example episode_014",
    )
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="print the discovered episodes and exit without opening a window",
    )
    args = parser.parse_args()
    args.data_root = args.data_root.expanduser().resolve()
    if args.fps <= 0:
        parser.error("--fps must be positive")
    return args


def image_paths(episode_dir: Path) -> list[Path]:
    """Return images in recorded sample order, falling back to filename order."""
    samples_path = episode_dir / "samples.jsonl"
    paths: list[Path] = []
    if samples_path.is_file():
        for line_number, line in enumerate(
            samples_path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                print(
                    f"[playback] warning: {samples_path}:{line_number}: {exc}",
                    file=sys.stderr,
                )
                continue
            relative = row.get("image_path")
            if isinstance(relative, str):
                path = episode_dir / relative
                if path.is_file():
                    paths.append(path)
    if paths:
        return paths
    return sorted((episode_dir / "images").glob("*.jpg"))


def discover_episodes(data_root: Path) -> list[tuple[Path, list[Path]]]:
    episodes: list[tuple[Path, list[Path]]] = []
    for episode_dir in sorted(data_root.glob("episode_*")):
        if not episode_dir.is_dir():
            continue
        paths = image_paths(episode_dir)
        if paths:
            episodes.append((episode_dir, paths))
    return episodes


def draw_overlay(
    frame,
    episode_name: str,
    episode_number: int,
    episode_count: int,
    frame_number: int,
    frame_count: int,
    status: str,
):
    display = frame.copy()
    lines = (
        f"{episode_name}  ({episode_number}/{episode_count})",
        f"frame {frame_number}/{frame_count}",
        status,
    )
    y = 28
    for line in lines:
        cv2.putText(
            display,
            line,
            (14, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (0, 0, 0),
            4,
            cv2.LINE_AA,
        )
        cv2.putText(
            display,
            line,
            (14, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (80, 255, 80),
            2,
            cv2.LINE_AA,
        )
        y += 27
    return display


def wait_key_until(deadline: float) -> int:
    """Keep the OpenCV window responsive until the next frame deadline."""
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return -1
        key = cv2.waitKey(max(1, min(10, int(remaining * 1000)))) & 0xFF
        if key != 255:
            return key


def main() -> int:
    args = parse_args()
    if not args.data_root.is_dir():
        print(f"[playback] data root not found: {args.data_root}", file=sys.stderr)
        return 2

    episodes = discover_episodes(args.data_root)
    if not episodes:
        print(f"[playback] no playable episodes under {args.data_root}", file=sys.stderr)
        return 2

    if args.start_episode is not None:
        names = [episode_dir.name for episode_dir, _ in episodes]
        if args.start_episode not in names:
            print(
                f"[playback] start episode not found: {args.start_episode}",
                file=sys.stderr,
            )
            return 2
        episodes = episodes[names.index(args.start_episode) :]

    total_frames = sum(len(paths) for _, paths in episodes)
    print(
        f"[playback] {len(episodes)} episodes, {total_frames} frames, "
        f"fps={args.fps:g}",
        flush=True,
    )
    for episode_dir, paths in episodes:
        print(f"  {episode_dir.name}: {len(paths)} frames")
    if args.list_only:
        return 0

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    episode_index = 0
    frame_index = 0
    period = 1.0 / args.fps
    next_deadline = time.monotonic()

    try:
        while episode_index < len(episodes):
            episode_dir, paths = episodes[episode_index]
            path = paths[frame_index]
            frame = cv2.imread(str(path))
            if frame is None:
                print(f"[playback] warning: cannot read {path}", file=sys.stderr)
                frame_index += 1
                if frame_index >= len(paths):
                    frame_index = len(paths) - 1
                continue

            at_end = frame_index == len(paths) - 1
            status = (
                "END: SPACE next | R replay | Q quit"
                if at_end
                else "SPACE next | P pause | R replay | Q quit"
            )
            display = draw_overlay(
                frame,
                episode_dir.name,
                episode_index + 1,
                len(episodes),
                frame_index + 1,
                len(paths),
                status,
            )
            cv2.imshow(WINDOW_NAME, display)

            if at_end:
                key = cv2.waitKey(0) & 0xFF
            else:
                next_deadline += period
                key = wait_key_until(next_deadline)

            if key in (ord("q"), ord("Q"), 27):
                break
            if key == ord(" "):
                episode_index += 1
                frame_index = 0
                next_deadline = time.monotonic()
                continue
            if key in (ord("r"), ord("R")):
                frame_index = 0
                next_deadline = time.monotonic()
                continue
            if key in (ord("p"), ord("P")) and not at_end:
                paused = True
                while paused:
                    pause_key = cv2.waitKey(0) & 0xFF
                    if pause_key in (ord("q"), ord("Q"), 27):
                        return 0
                    if pause_key == ord(" "):
                        episode_index += 1
                        frame_index = 0
                        paused = False
                    elif pause_key in (ord("r"), ord("R")):
                        frame_index = 0
                        paused = False
                    elif pause_key in (ord("p"), ord("P")):
                        paused = False
                next_deadline = time.monotonic()
                continue

            if at_end:
                continue
            frame_index += 1

        print("[playback] finished", flush=True)
        return 0
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    raise SystemExit(main())
