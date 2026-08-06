"""Live camera teleop for the HumanEgo two-finger LinkerHand MVP.

This command is deliberately sim-first.  It reads hand landmarks from an
existing perception source, runs the normal LinkerHand retargeter, then locks
the command down to thumb + index before the safety filter and PyBullet mirror.

Examples
--------
Use a webcam to control thumb + index in the sim:

    .venv/bin/python -m src.humanego_linkerhand.camera_two_finger \
      --source webcam --camera-index 0 --side right --show-camera

Run a headless smoke test with the committed replay fixture:

    .venv/bin/python -m src.humanego_linkerhand.camera_two_finger \
      --source replay --headless --max-frames 5
"""
from __future__ import annotations

import argparse
import time
from typing import Iterator, Optional, Tuple

import numpy as np

from .two_finger import lock_candidate_to_two_finger

DEFAULT_DT = 1.0 / 30.0


def _build_source(args: argparse.Namespace):
    if args.source == "webcam":
        from src.perception.mediapipe_source import MediaPipeHandSource
        return MediaPipeHandSource(
            camera_index=args.camera_index,
            fps=args.rate,
            fingertip_extend=args.fingertip_extend,
            fingertip_lateral=args.fingertip_lateral,
            fingertip_straighten=args.fingertip_straighten,
        )
    if args.source == "realsense":
        from src.perception.realsense_source import RealSenseHandSource
        return RealSenseHandSource(fps=int(round(args.rate)))
    if args.source == "video":
        from src.perception.video_source import VideoHandSource
        if not args.video_path:
            raise ValueError("--source video requires --video-path")
        return VideoHandSource(
            args.video_path,
            fps=args.rate,
            playback_rate=args.playback_rate,
            fingertip_extend=args.fingertip_extend,
            fingertip_lateral=args.fingertip_lateral,
            fingertip_straighten=args.fingertip_straighten,
        )
    raise ValueError(f"camera source not built for {args.source!r}")


def _live_stream(args: argparse.Namespace, source, model: L20VizModel
                 ) -> Iterator[Tuple[str, np.ndarray, float]]:
    from src.perception.pipeline import HandPipeline
    from src.perception.one_euro import OneEuroConfig
    from src.viz.app import _camera_preview

    pipeline = HandPipeline(
        source,
        smoothing=not args.no_smoothing,
        one_euro=OneEuroConfig(
            min_cutoff=args.one_euro_min_cutoff,
            beta=args.one_euro_beta,
            d_cutoff=args.one_euro_d_cutoff,
        ),
        image_mirrored=args.image_mirrored,
        force_side=args.side,
    )
    cv2 = None
    if args.show_camera:
        import cv2 as _cv2
        cv2 = _cv2

    n = 0
    for det in source:
        pf = pipeline.process(det)
        if pf is None:
            if not _camera_preview(cv2, source, "two-finger camera", None):
                break
            continue
        if pf.side != model.side:
            model.side = pf.side
        if not _camera_preview(cv2, source, "two-finger camera", pf):
            break
        yield pf.side, pf.landmarks, pf.t
        n += 1
        if args.max_frames is not None and n >= args.max_frames:
            break
    if cv2 is not None:
        cv2.destroyAllWindows()


def _replay_stream(args: argparse.Namespace) -> Iterator[Tuple[str, np.ndarray, float]]:
    from src.viz.app import replay_stream

    stream = replay_stream(args.side)
    for i, item in enumerate(stream):
        if args.max_frames is not None and i >= args.max_frames:
            break
        yield item
        if args.realtime:
            time.sleep(1.0 / args.rate if args.rate > 0 else DEFAULT_DT)


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", choices=("webcam", "realsense", "video", "replay"),
                    default="webcam")
    ap.add_argument("--camera-index", type=int, default=0)
    ap.add_argument("--video-path")
    ap.add_argument("--playback-rate", type=float, default=1.0)
    ap.add_argument("--side", choices=("right", "left"), default="right",
                    help="force physical LinkerHand side")
    ap.add_argument("--image-mirrored", action="store_true")
    ap.add_argument("--rate", type=float, default=30.0)
    ap.add_argument("--max-frames", type=int)
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--show-camera", action="store_true")
    ap.add_argument("--realtime", action="store_true",
                    help="sleep during replay to approximate camera rate")
    ap.add_argument("--no-filter", action="store_true",
                    help="bypass safety.filter after the two-finger lock")
    ap.add_argument("--no-smoothing", action="store_true",
                    help="disable perception-side one-euro smoothing")
    ap.add_argument("--one-euro-min-cutoff", type=float, default=1.5,
                    help="perception One Euro min cutoff in Hz; lower is smoother but laggier")
    ap.add_argument("--one-euro-beta", type=float, default=0.05,
                    help="perception One Euro speed coefficient; higher reduces lag while moving")
    ap.add_argument("--one-euro-d-cutoff", type=float, default=1.0,
                    help="perception One Euro derivative cutoff in Hz")
    ap.add_argument("--fingertip-extend", default="0.0")
    ap.add_argument("--fingertip-lateral", default="0.0")
    ap.add_argument("--fingertip-straighten", default="0.0")
    ap.add_argument("--thumb-gain", type=float, default=1.0)
    ap.add_argument("--thumb-cross-gain", type=float, default=0.28)
    ap.add_argument("--thumb-assist-smooth", type=float, default=0.72)
    ap.add_argument("--thumb-grasp-gain", type=float, default=0.38)
    ap.add_argument("--thumb-tip-gain", type=float, default=1.12)
    ap.add_argument("--thumb-orient-gain", type=float, default=0.65)
    ap.add_argument("--print-every", type=int, default=15,
                    help="print active thumb/index radians every N frames; 0 disables")
    return ap


def main(argv: Optional[list[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    from src.viz.core import drive
    from src.viz.render import L20VizModel
    from src.viz.app import (
        _compose_adjust,
        _thumb_adjuster,
        _thumb_grasp_adjuster,
        _thumb_orient_adjuster,
        _thumb_tip_adjuster,
    )

    model = L20VizModel(args.side, gui=not args.headless)
    source = None
    records = []

    def _on_record(rec: dict, out: dict) -> None:
        if args.print_every <= 0 or rec["frame"] % args.print_every != 0:
            return
        q = rec["command"]
        active = [round(float(q[i]), 3) for i in (0, 5, 10, 15, 1, 6, 16)]
        print(f"[camera-two-finger] frame={rec['frame']} side={rec['side']} q={active}",
              flush=True)

    adjust = _compose_adjust(
        _thumb_adjuster(args.thumb_gain, args.thumb_cross_gain,
                        args.thumb_assist_smooth),
        _thumb_grasp_adjuster(args.thumb_grasp_gain),
        _thumb_orient_adjuster(args.thumb_orient_gain),
        _thumb_tip_adjuster(args.thumb_tip_gain),
        lock_candidate_to_two_finger,
    )

    try:
        if args.source == "replay":
            stream = _replay_stream(args)
        else:
            source = _build_source(args)
            stream = _live_stream(args, source, model)
        records = drive(
            model,
            stream,
            use_filter=not args.no_filter,
            dt=1.0 / args.rate if args.rate > 0 else DEFAULT_DT,
            candidate_adjust=adjust,
            on_record=_on_record,
        )
    finally:
        if source is not None:
            source.close()
        model.close()
    print(f"[camera-two-finger] done frames={len(records)}", flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
