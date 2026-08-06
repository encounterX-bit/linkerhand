"""Loop hand clips through the viz sim, with the source video shown alongside.

Side-by-side: a PyBullet GUI window (the L20 mirror, driven by the real
``core.drive`` loop) + an OpenCV window playing the source clip with a small
status overlay (clip / side / detected-or-held / frame). Pass one or more clips;
it plays each once and **auto-advances** to the next, cycling the whole list
forever until you press ``q`` in the video window (or Ctrl-C). ``--laps N`` stops
after N full passes over the list.

Adds no module logic — orchestrates VideoHandSource / HandPipeline / core.drive /
L20VizModel via their public APIs. Two correctness points:

* **Handedness / chirality.** ``L20VizModel`` loads a left- *or* right-hand URDF at
  construction and does NOT swap it later, so the model must be BUILT for the
  clip's side or the sim shows the mirrored hand. We resolve each clip's side up
  front (first detected frame -> ``to_l20_side``, or ``--side``), build the model
  for that side, and force the pipeline to the same side so video and sim agree.
  The model is rebuilt only when the next clip needs the other hand.
* **Frame alignment.** The displayed clip is decoded by a *separate*
  ``cv2.VideoCapture`` advanced in lockstep with the detector's own frame counter,
  so the video and the sim stay frame-aligned without re-running MediaPipe.

    python scripts/loop_video_with_view.py assets/video/normalized/*.mp4
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.viz.core import DEFAULT_DT, drive  # noqa: E402
from src.viz.render import L20VizModel  # noqa: E402
from src.perception.handedness import to_l20_side  # noqa: E402
from src.perception.pipeline import HandPipeline  # noqa: E402
from src.perception.video_source import VideoHandSource  # noqa: E402

WIN = "source video (q quits)"


def resolve_side(path, *, force=None, image_mirrored=False, probe_frames=60):
    """Physical L20 side for a clip: forced, else first detected frame's hand.

    Reads at most ``probe_frames`` frames looking for the first detection with a
    handedness label, maps it via ``to_l20_side`` (the same mapping validation
    uses), and falls back to 'right' if nothing detects. Cheap: stops at the first
    hit. The detection pass is throwaway; the real run re-opens the clip at frame 0.
    """
    if force is not None:
        return force
    src = VideoHandSource(path)
    try:
        for _ in range(probe_frames):
            try:
                det = src.read()
            except StopIteration:
                break
            if det.ok and det.handedness is not None:
                return to_l20_side(det.handedness, image_mirrored)
    finally:
        src.close()
    return "right"


def _overlay(frame, clip, pf, lap):
    h = frame.shape[0]
    scale = max(0.5, h / 900.0)
    state = "HELD" if pf.held else "DET"
    color = (0, 200, 0) if pf.detected else (0, 165, 255)
    for i, line in enumerate((
        f"{clip}  lap {lap}  frame {int(round(pf.t * 30))}",
        f"side={pf.side}  {state}  score={pf.score:.2f}",
    )):
        cv2.putText(frame, line, (12, int(28 * scale) + i * int(30 * scale)),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, line, (12, int(28 * scale) + i * int(30 * scale)),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def play_once(model, path, *, side, image_mirrored, use_filter, lap, view_max, stop):
    """One pass of a single clip through the sim + the side-by-side video window."""
    clip = os.path.basename(path)
    source = VideoHandSource(path)
    pipeline = HandPipeline(source, smoothing=True, image_mirrored=image_mirrored,
                            force_side=side)
    disp = cv2.VideoCapture(path)
    period = source.frame_period
    disp_idx = 0

    def stream():
        nonlocal disp_idx
        for pf in pipeline.run():
            target = getattr(source, "_frame_idx", disp_idx + 1)
            frame = None
            while disp_idx < target:
                ok, f = disp.read()
                disp_idx += 1
                if ok:
                    frame = f
            if frame is not None:
                if frame.shape[0] > view_max:
                    s = view_max / frame.shape[0]
                    frame = cv2.resize(frame, None, fx=s, fy=s)
                _overlay(frame, clip, pf, lap)
                cv2.imshow(WIN, frame)
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    stop["q"] = True
            yield pf.side, pf.landmarks, pf.t
            time.sleep(period)
            if stop["q"]:
                break

    try:
        drive(model, stream(), use_filter=use_filter, dt=DEFAULT_DT)
    finally:
        disp.release()
        source.close()


def run(paths, *, side=None, image_mirrored=False, use_filter=True, laps=None,
        view_max=720):
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.moveWindow(WIN, 20, 40)

    # Resolve every clip's side once; build the model for the first clip's hand.
    sides = [resolve_side(p, force=side, image_mirrored=image_mirrored)
             for p in paths]
    for p, s in zip(paths, sides):
        print(f"  {os.path.basename(p)} -> sim hand: {s}", flush=True)

    model = L20VizModel(sides[0], gui=True)
    stop = {"q": False}
    lap = 0
    try:
        while not stop["q"] and (laps is None or lap < laps):
            lap += 1
            for p, want in zip(paths, sides):
                if stop["q"]:
                    break
                if model.side != want:          # next clip is the other hand
                    model.close()
                    model = L20VizModel(want, gui=True)
                print(f">>> [lap {lap}] {os.path.basename(p)}  (sim={want})",
                      flush=True)
                play_once(model, p, side=want, image_mirrored=image_mirrored,
                          use_filter=use_filter, lap=lap, view_max=view_max,
                          stop=stop)
    except KeyboardInterrupt:
        pass
    finally:
        model.close()
        cv2.destroyAllWindows()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("clips", nargs="+", help="one or more clips (cycles through all)")
    ap.add_argument("--side", choices=("right", "left"), default=None,
                    help="force the hand for ALL clips (default: resolve per clip "
                         "from MediaPipe handedness)")
    ap.add_argument("--image-mirrored", action="store_true",
                    help="clips are selfie/front-camera (mirrored) -> swap handedness")
    ap.add_argument("--no-filter", dest="use_filter", action="store_false")
    ap.add_argument("--laps", type=int, default=None,
                    help="stop after N full passes over the clip list")
    args = ap.parse_args(argv)
    run(args.clips, side=args.side, image_mirrored=args.image_mirrored,
        use_filter=args.use_filter, laps=args.laps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
