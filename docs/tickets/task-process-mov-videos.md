# Task: process new MOV hand videos through the sim

**Fresh session — orient first.** You have no prior context. Before anything, read
`STATE.md` and `CLAUDE.md`, then confirm these already exist:
`src/perception/video_source.py` (`VideoHandSource`), the `src/viz/` GUI loop with a
`--source video --video-path` option, `src/kinematics/`, `finger_retarget.retarget()`,
`safety.filter()`. Build on them; change tested modules only if forced, and additively.

**Owns:** a small MOV-normalization utility + a runner/report. No new gates.
**Gate:** none — sim-only, hardware-free (no comms, no HW_ENABLE_TOKEN).
**Input:** the user's `.MOV` files (take a directory/glob argument; these are the
user's own footage, so no licensing concern).
**Done =** each usable MOV streams through the loop into the GUI, with per-clip
detection rate reported and a capture saved.

---

## What this is
The user's own hand videos driving the sim hand through the existing
video → MediaPipe → retarget → filter → sim pipeline. Still monocular RGB (weak z):
this tests the front end + plumbing on real footage, not depth accuracy (that's the
RealSense path). Soft thumb / mushy depth-axis motion is the monocular limit, expected.

## The MOV gotchas (handle these or detection silently dies)
1. **Rotation metadata.** Phone MOVs are often recorded sideways with a rotation flag.
   OpenCV's `VideoCapture` historically ignores it, so frames arrive rotated 90° and
   MediaPipe sees a sideways hand → near-zero detection. If detection is ~0, suspect
   this first.
2. **Codec.** MOV is a container; iPhone defaults to **HEVC/H.265**, which some
   OpenCV/FFmpeg builds can't decode (silent empty frames).

**Recommended fix — normalize once, up front, leave `VideoHandSource` untouched:** if
`ffmpeg` is available, transcode each MOV → standard H.264 MP4, upright (ffmpeg
auto-applies rotation), downscaled to ~720p and ~30 fps (MediaPipe needs nothing more,
and it's faster):
`ffmpeg -i in.mov -vf "scale=-2:720" -r 30 -c:v libx264 -pix_fmt yuv420p out.mp4`
Then feed `out.mp4` to the existing loop. If `ffmpeg` isn't present, handle rotation
and codec inside the reader instead — but prefer the normalize step.

## Steps
1. Orient (above). Locate the MOV files from the provided path.
2. Normalize each MOV (ffmpeg → upright 720p/30fps H.264 MP4), or report if ffmpeg
   is missing and fall back to in-reader handling.
3. **Validate detection before streaming:** run MediaPipe over each normalized clip,
   report per-frame hand-detection rate + which hand. If a clip is near-zero, re-check
   rotation before discarding.
4. Stream each clip with good detection through the loop
   (`--source video --video-path <mp4>`) into the PyBullet GUI; save a capture
   (gif/mp4/frame dump) per clip to the viz `out/` dir.
5. Handedness check per clip (`to_l20_side`; phone front-camera clips are mirrored).
6. Report in `STATE.md`: per-clip detection rate, which hand, normalization applied,
   which tracked cleanly, observed thumb/depth softness, any mirror fix.

## Guardrails
- Success must be **evidenced** — real detection numbers + saved captures, not asserted.
- Keep large normalized files out of git (gitignore the work dir).
- Don't modify `retarget`, `safety.filter`, `src/kinematics`, or the loop core — this is
  preprocessing + running.

## Context to load
`STATE.md`, `CLAUDE.md`, `src/perception/video_source.py`, `src/viz/` (loop + options),
this task.
