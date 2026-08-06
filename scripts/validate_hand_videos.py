"""Validate fetched hand clips against MediaPipe BEFORE streaming them.

Runs the *real* VideoHandSource (the same MediaPipe RGB path the viz loop uses)
over each clip and reports the per-frame hand-detection rate + which hand it
sees. This is the "bad clip vs broken pipeline" gate from the ticket: clips that
detect below ``--min-rate`` (default 0.70) are flagged DISCARD.

Pure reporting tool — imports VideoHandSource read-only, changes no module logic.

    python scripts/validate_hand_videos.py assets/video/*.webm
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.perception.handedness import to_l20_side  # noqa: E402
from src.perception.video_source import VideoHandSource  # noqa: E402


def validate(path: str, image_mirrored: bool = False) -> dict:
    src = VideoHandSource(path)
    total = detected = 0
    labels = collections.Counter()
    scores = []
    try:
        while True:
            try:
                det = src.read()
            except StopIteration:
                break
            total += 1
            if det.ok:
                detected += 1
                scores.append(det.score)
                if det.handedness is not None:
                    labels[det.handedness] += 1
    finally:
        src.close()

    rate = detected / total if total else 0.0
    cam_label = labels.most_common(1)[0][0] if labels else None
    l20 = to_l20_side(cam_label, image_mirrored) if cam_label else None
    return {
        "path": path,
        "native_fps": src.native_fps,
        "frame_count_meta": src.frame_count,
        "frames_read": total,
        "frames_detected": detected,
        "detection_rate": round(rate, 4),
        "camera_label_counts": dict(labels),
        "camera_label_majority": cam_label,
        "l20_side": l20,  # physical side via to_l20_side (image_mirrored=%s)
        "mean_score": round(sum(scores) / len(scores), 4) if scores else 0.0,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("clips", nargs="+", help="video file paths (globs ok)")
    ap.add_argument("--min-rate", type=float, default=0.70,
                    help="discard clips below this per-frame detection rate")
    ap.add_argument("--image-mirrored", action="store_true",
                    help="treat clips as selfie/mirrored when mapping handedness")
    ap.add_argument("--out", default=None, help="write JSON report here")
    args = ap.parse_args(argv)

    paths = []
    for c in args.clips:
        paths.extend(sorted(glob.glob(c)) or [c])

    reports = []
    for p in paths:
        print(f"[validate] {p} ...", flush=True)
        r = validate(p, image_mirrored=args.image_mirrored)
        r["verdict"] = "KEEP" if r["detection_rate"] >= args.min_rate else "DISCARD"
        reports.append(r)
        print(f"  rate={r['detection_rate']:.1%} "
              f"({r['frames_detected']}/{r['frames_read']})  "
              f"cam={r['camera_label_majority']} -> l20={r['l20_side']}  "
              f"score={r['mean_score']:.3f}  => {r['verdict']}", flush=True)

    if args.out:
        with open(args.out, "w") as f:
            json.dump({"min_rate": args.min_rate,
                       "image_mirrored": args.image_mirrored,
                       "clips": reports}, f, indent=2)
        print(f"[report] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
