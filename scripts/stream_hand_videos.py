"""Stream a validated hand clip through the existing viz loop and save a capture.

Builds the *real* components the viz ``run_video`` path uses —
``VideoHandSource`` -> ``HandPipeline`` -> ``core.drive`` (retarget -> filter ->
``L20VizModel.set_joints``) — and hangs a frame-grabber on ``drive``'s
``on_record`` hook to dump a GIF of the PyBullet sim per clip. Adds no module
logic; it only orchestrates the existing pieces and renders an artifact.

    python scripts/stream_hand_videos.py assets/video/woman_counting_on_fingers.webm \
        --out-dir tests/viz/out --gui

By default it shows the PyBullet GUI window (DISPLAY permitting) AND writes the
GIF via the CPU tiny-renderer (independent of the window), so the artifact is
produced even headless. ``--no-gui`` forces DIRECT.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pybullet as pb
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.viz.core import DEFAULT_DT, drive  # noqa: E402
from src.viz.render import L20VizModel  # noqa: E402
from src.perception.handedness import to_l20_side  # noqa: E402
from src.perception.pipeline import HandPipeline  # noqa: E402
from src.perception.video_source import VideoHandSource  # noqa: E402


def resolve_side(path, *, image_mirrored=False, probe_frames=60):
    """Physical L20 side from the first detected frame (else 'right').

    ``L20VizModel`` loads a left/right-hand URDF at construction and never swaps
    it, so the model MUST be built for the clip's side or the capture shows the
    mirrored hand. Cheap throwaway pass; the real run re-opens the clip at 0.
    """
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


def _grab(model, w: int, h: int) -> np.ndarray:
    """CPU (tiny-renderer) RGB grab of the sim, independent of any GUI window."""
    view = pb.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=[0.0, 0.0, 0.08], distance=0.35,
        yaw=50, pitch=-25, roll=0, upAxisIndex=2)
    proj = pb.computeProjectionMatrixFOV(fov=55, aspect=w / h, nearVal=0.01,
                                         farVal=2.0)
    _, _, rgba, _, _ = pb.getCameraImage(
        w, h, viewMatrix=view, projectionMatrix=proj,
        renderer=pb.ER_TINY_RENDERER, physicsClientId=model.cid)
    return np.reshape(np.asarray(rgba, dtype=np.uint8), (h, w, 4))[:, :, :3]


def stream(path: str, out_dir: str, *, gui: bool = True, image_mirrored: bool = False,
           every: int = 2, size: int = 480, max_frames=None) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    # Build the model for the clip's actual hand (the URDF chirality is fixed at
    # construction); force the pipeline to the same side so cmd + render agree.
    side = resolve_side(path, image_mirrored=image_mirrored)
    source = VideoHandSource(path)
    pipeline = HandPipeline(source, smoothing=True, image_mirrored=image_mirrored,
                            force_side=side)
    model = L20VizModel(side, gui=gui)

    frames = []
    sides = []

    def on_record(rec, out):
        sides.append(rec["side"])
        if rec["frame"] % every == 0:
            frames.append(Image.fromarray(_grab(model, size, size)))

    def _stream():
        n = 0
        for pf in pipeline.run():
            if pf.side != model.side:
                model.side = pf.side
            yield pf.side, pf.landmarks, pf.t
            n += 1
            if max_frames is not None and n >= max_frames:
                break

    try:
        records = drive(model, _stream(), use_filter=True, dt=DEFAULT_DT,
                        on_record=on_record)
    finally:
        source.close()
        model.close()

    base = os.path.splitext(os.path.basename(path))[0]
    gif_path = os.path.join(out_dir, f"sim_{base}.gif")
    png_path = os.path.join(out_dir, f"sim_{base}_last.png")
    if frames:
        frames[0].save(gif_path, save_all=True, append_images=frames[1:],
                       duration=int(1000 * every / source.fps), loop=0)
        frames[-1].save(png_path)

    n_mod = sum(1 for r in records if r["modified"])
    side_maj = max(set(sides), key=sides.count) if sides else None
    return {
        "path": path,
        "frames_streamed": len(records),
        "gif": gif_path if frames else None,
        "gif_frames": len(frames),
        "last_png": png_path if frames else None,
        "filter_modified_frames": n_mod,
        "side_majority": side_maj,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("clips", nargs="+")
    ap.add_argument("--out-dir", default="tests/viz/out")
    ap.add_argument("--no-gui", dest="gui", action="store_false")
    ap.add_argument("--image-mirrored", action="store_true")
    ap.add_argument("--every", type=int, default=2, help="capture every Nth frame")
    ap.add_argument("--size", type=int, default=480)
    ap.add_argument("--max-frames", type=int, default=None)
    args = ap.parse_args(argv)

    for c in args.clips:
        print(f"[stream] {c} ...", flush=True)
        r = stream(c, args.out_dir, gui=args.gui, image_mirrored=args.image_mirrored,
                   every=args.every, size=args.size, max_frames=args.max_frames)
        print(f"  streamed {r['frames_streamed']} frames, side={r['side_majority']}, "
              f"filter modified {r['filter_modified_frames']}, "
              f"gif={r['gif']} ({r['gif_frames']} frames)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
