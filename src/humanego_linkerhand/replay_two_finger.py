"""Replay HumanEgo-style two-finger actions on LinkerHand sim or JSONL output.

Examples
--------
Run the built-in thumb-index pinch demo in PyBullet:

    python -m src.humanego_linkerhand.replay_two_finger --show-sim

Convert a policy action JSONL to canonical 20-joint commands:

    python -m src.humanego_linkerhand.replay_two_finger \
      --trajectory actions.jsonl --mode joint7 --out-jsonl out/l20_cmds.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Iterable

from .two_finger import (
    ACTION_DIMS,
    TwoFingerConfig,
    action_to_l20,
    demo_actions,
    iter_action_records,
)


def _iter_actions(args: argparse.Namespace) -> Iterable[tuple[list[float], float | None]]:
    if args.trajectory:
        yield from iter_action_records(args.trajectory)
        return
    for action in demo_actions(args.mode, frames=args.demo_frames):
        yield action, None


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trajectory",
                    help="JSON/JSONL/NPY/NPZ policy action trajectory; omitted runs demo")
    ap.add_argument("--mode", choices=tuple(ACTION_DIMS), default="pinch3")
    ap.add_argument("--input-range", choices=("normalized", "radians"),
                    default="normalized",
                    help="joint7 action interpretation; pinch3 is always normalized")
    ap.add_argument("--side", choices=("right", "left"), default="right")
    ap.add_argument("--rate", type=float, default=30.0)
    ap.add_argument("--demo-frames", type=int, default=90)
    ap.add_argument("--show-sim", action="store_true")
    ap.add_argument("--out-jsonl",
                    help="write canonical l20_targets-like records here")
    ap.add_argument("--print-every", type=int, default=30,
                    help="print one compact status line every N frames; 0 disables")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.mode == "pinch3" and args.input_range == "radians":
        print("pinch3 expects normalized actions; use --mode joint7 for radians",
              file=sys.stderr)
        return 2

    cfg = TwoFingerConfig(
        side=args.side,
        mode=args.mode,
        input_range=args.input_range,
    )

    model = None
    out_f = None
    try:
        if args.show_sim:
            from src.viz.render import L20VizModel
            model = L20VizModel(args.side, gui=True)
        if args.out_jsonl:
            out_path = Path(args.out_jsonl)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_f = out_path.open("w", encoding="utf-8")

        dt = 1.0 / args.rate if args.rate > 0 else 0.0
        for frame_i, (action, t) in enumerate(_iter_actions(args)):
            target = action_to_l20(action, cfg)
            target["frame"] = frame_i
            target["t"] = float(frame_i * dt if t is None else t)
            target["action"] = [float(v) for v in action]
            if model is not None:
                model.set_joints(target["joint_rad"])
            if out_f is not None:
                out_f.write(json.dumps(target, separators=(",", ":")) + "\n")
            if args.print_every > 0 and frame_i % args.print_every == 0:
                q = target["joint_rad"]
                active = [round(q[i], 3) for i in (0, 5, 10, 15, 1, 6, 16)]
                print(f"[two-finger] frame={frame_i} action={target['action']} q={active}",
                      flush=True)
            if model is not None and dt > 0:
                time.sleep(dt)
        return 0
    finally:
        if out_f is not None:
            out_f.close()
        if model is not None:
            model.close()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
