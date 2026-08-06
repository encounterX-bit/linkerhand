"""Normalize phone .MOV clips to upright 720p/30fps H.264 MP4 before streaming.

Phone MOVs hit two gotchas that silently kill MediaPipe detection (see the task
ticket): they are usually **HEVC/H.265** (some OpenCV/FFmpeg builds emit empty
frames) and carry a **rotation flag** that ``cv2.VideoCapture`` ignores, so frames
arrive sideways and MediaPipe sees a rotated hand. The robust fix is to transcode
once, up front, and let ``VideoHandSource`` stay untouched.

This wraps the ticket's recommended ffmpeg command:

    ffmpeg -i in.mov -vf "scale=-2:720" -r 30 -c:v libx264 -pix_fmt yuv420p out.mp4

ffmpeg **auto-applies the rotation metadata**, so the output is upright H.264; the
``scale=-2:720`` keeps the height at 720 (width auto, even) and ``-r 30`` pins
30 fps. Pure preprocessing — it touches no module logic and writes only the work
dir (gitignored).

    python scripts/normalize_mov_videos.py assets/video/*.MOV \
        --out-dir assets/video/normalized
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys


def probe(path: str) -> dict:
    """Return codec/dims/fps/rotation for the first video stream (best-effort)."""
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries",
        "stream=codec_name,width,height,r_frame_rate,nb_frames",
        "-show_entries", "stream_side_data=rotation",
        "-of", "json", path,
    ]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
        data = json.loads(out)
        st = (data.get("streams") or [{}])[0]
        rot = None
        for sd in st.get("side_data_list", []) or []:
            if "rotation" in sd:
                rot = sd["rotation"]
        return {
            "codec": st.get("codec_name"),
            "width": st.get("width"),
            "height": st.get("height"),
            "r_frame_rate": st.get("r_frame_rate"),
            "nb_frames": st.get("nb_frames"),
            "rotation": rot,
        }
    except Exception as exc:  # pragma: no cover - probe is advisory only
        return {"error": str(exc)}


def normalize(path: str, out_dir: str, *, height: int = 720, fps: int = 30,
              overwrite: bool = False) -> dict:
    """Transcode one MOV -> upright H.264 MP4 in ``out_dir``. Returns a report."""
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(path))[0]
    out_path = os.path.join(out_dir, f"{base}.mp4")

    src_info = probe(path)
    if os.path.isfile(out_path) and not overwrite:
        return {"src": path, "out": out_path, "skipped": True,
                "src_info": src_info, "out_info": probe(out_path)}

    cmd = [
        "ffmpeg", "-y", "-i", path,
        "-vf", f"scale=-2:{height}",
        "-r", str(fps),
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-an",  # drop audio; the pipeline is video-only
        out_path,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    ok = proc.returncode == 0 and os.path.isfile(out_path)
    return {
        "src": path,
        "out": out_path if ok else None,
        "skipped": False,
        "ok": ok,
        "src_info": src_info,
        "out_info": probe(out_path) if ok else None,
        "ffmpeg_err": None if ok else proc.stderr[-800:],
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("clips", nargs="+", help="MOV file paths (globs ok)")
    ap.add_argument("--out-dir", default="assets/video/normalized")
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--out", default=None, help="write a JSON report here")
    args = ap.parse_args(argv)

    if shutil.which("ffmpeg") is None:
        print("ERROR: ffmpeg not found on PATH. Install it, or handle rotation/codec "
              "inside the reader instead (see ticket).", file=sys.stderr)
        return 2

    paths = []
    for c in args.clips:
        paths.extend(sorted(glob.glob(c)) or [c])

    reports = []
    for p in paths:
        print(f"[normalize] {p} ...", flush=True)
        r = normalize(p, args.out_dir, height=args.height, fps=args.fps,
                      overwrite=args.overwrite)
        reports.append(r)
        si, oi = r.get("src_info", {}), r.get("out_info") or {}
        tag = "SKIP (exists)" if r.get("skipped") else ("OK" if r.get("ok") else "FAIL")
        print(f"  {si.get('codec')} {si.get('width')}x{si.get('height')} "
              f"rot={si.get('rotation')} -> {oi.get('codec')} "
              f"{oi.get('width')}x{oi.get('height')} rot={oi.get('rotation')}  "
              f"=> {tag}", flush=True)
        if not r.get("skipped") and not r.get("ok"):
            print(r.get("ffmpeg_err"), file=sys.stderr)

    if args.out:
        with open(args.out, "w") as f:
            json.dump({"out_dir": args.out_dir, "clips": reports}, f, indent=2)
        print(f"[report] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
