#!/usr/bin/env python3
"""Leave-one-take-out validation for a MediaPipe action library."""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from src.comms.action_library import ActionLibrary, dtw_distance


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--margin", type=float, default=0.015)
    parser.add_argument("--minimum-accuracy", type=float, default=0.90)
    return parser.parse_args(argv)


def validate(library: ActionLibrary, margin: float) -> tuple[int, int, Counter]:
    correct = 0
    total = 0
    confusions: Counter = Counter()
    for expected in library.primitives.values():
        if len(expected.templates) < 2:
            print(
                f"[validate] SKIP {expected.id}:{expected.name}: need at least two takes",
                file=sys.stderr,
            )
            continue
        for held_index, held in enumerate(expected.templates):
            scores: list[tuple[float, int, str, float]] = []
            for candidate in library.primitives.values():
                candidates = [
                    template
                    for index, template in enumerate(candidate.templates)
                    if candidate.id != expected.id or index != held_index
                ]
                if candidates:
                    score = min(dtw_distance(held, template) for template in candidates)
                    scores.append((score, candidate.id, candidate.name, candidate.threshold))
            scores.sort()
            if not scores:
                continue
            best = scores[0]
            second_distance = scores[1][0] if len(scores) > 1 else np.inf
            predicted = best[1] if best[0] <= best[3] and second_distance - best[0] >= margin else -1
            total += 1
            if predicted == expected.id:
                correct += 1
            else:
                confusions[(expected.id, predicted)] += 1
                print(
                    f"[validate] MISS expected={expected.id}:{expected.name} "
                    f"predicted={predicted}:{best[2] if predicted >= 0 else 'unknown'} "
                    f"best={best[0]:.4f} second={second_distance:.4f}",
                    flush=True,
                )
    return correct, total, confusions


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        library = ActionLibrary.load(args.library)
        correct, total, confusions = validate(library, max(0.0, args.margin))
    except (OSError, ValueError, KeyError) as exc:
        print(f"[validate] {exc}", file=sys.stderr)
        return 2
    if total == 0:
        print("[validate] no eligible takes; record at least two per primitive", file=sys.stderr)
        return 2
    accuracy = correct / total
    print(f"[validate] accuracy={accuracy:.3%} correct={correct}/{total}")
    for (expected, predicted), count in confusions.most_common():
        print(f"[validate] confusion {expected}->{predicted}: {count}")
    return 0 if accuracy >= args.minimum_accuracy else 1


if __name__ == "__main__":
    raise SystemExit(main())
