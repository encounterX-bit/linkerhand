# Amendment to `viz-agent`: VideoHandSource (recorded hand video → sim)

**Status:** Stage 1 (live RealSense + GUI loop) is already built. This is an additive
follow-up — a third input source for the existing `src/viz/` loop. **Do not rewrite
the loop or the RealSense path; both stay as-is.**
**Owns:** new `src/perception/video_source.py`; one additive option in the existing
`src/viz/` source selection.
**Gate:** none — same sim-only, hardware-free layer as the parent ticket.
**Runtime deps:** `mediapipe`, `opencv` (already documented for the RGB backend).
**Done =** the loop runs from a video file end-to-end, and the existing
pipeline-equivalence test still passes unchanged.

> **Orient first:** read `STATE.md` and confirm what Stage 1 produced — the `src/viz/`
> loop, `realsense_source.py`, and the replay/camera-free fallback. Build on those;
> change neither the loop's core nor the RealSense source.

---

## What this proves (and what it doesn't)
A plain hand video is **monocular RGB — no depth.** So this routes through MediaPipe's
estimated z (the weak-z path), NOT metric depth. It validates the **plumbing end to
end** — real (non-synthetic) hand motion → MediaPipe → palm-plane frame → `retarget()`
→ `safety.filter()` → sim joints — but it does **not** validate depth/retargeting
accuracy; that still belongs to the RealSense path. Set the depth-confidence signal
**LOW**, same as the existing RGB/MediaPipe backend. Treat green here as "the pipeline
carries a real hand through to the sim hand," not "the mapping is accurate."

## Build
### 1. `src/perception/video_source.py` (new `HandSource`)
- **Reuse the existing MediaPipe RGB detection path** — do not reimplement MediaPipe.
  This source differs from `MediaPipeHandSource` only in frame acquisition:
  `cv2.VideoCapture(path)` instead of a webcam.
- Per frame: detect → 21 landmarks → existing pipeline (palm-plane frame, one-euro
  smoothing) → `hand_landmarks` contract.
- Honor the video's native FPS, with an optional playback-rate override.
- No-detection / low-confidence frames: hold-last-good + flag (the pipeline path),
  never emit NaN. End-of-file → stop cleanly.
- Confidence LOW (monocular).

### 2. Wire into the existing `src/viz/` loop (additive only)
- Add a source option so the loop can be launched with the video source, e.g.
  `--source video --video-path <file>`, alongside the existing realsense / replay
  options. Everything downstream (retarget → filter → `resetJointState` + mimic
  enforcement from `src/kinematics` → GUI) is unchanged.

## Tests
- **VideoHandSource unit (no GUI):** on a short committed clip (or a synthetic frame
  stub), emits valid `hand_landmarks`, confidence LOW, handles a no-detection frame and
  EOF gracefully, honors FPS.
- **Equivalence unchanged:** the Stage-1 pipeline-equivalence test (viz loop ≡ existing
  pipeline on the same input) still passes — confirms the new source didn't perturb the
  loop.
- Live "watch the sim follow the video" is a manual visual check.

## Practical notes (these decide whether it looks good)
- MediaPipe z is hand-relative and noisy: a video where the hand is **large in frame,
  well-lit, palm-toward or side-on, moving slowly** tracks far better than a small/fast
  hand. The thumb and fine flexion will look soft — that's the monocular limit, not a
  bug, and it's exactly what RealSense later fixes.
- **Handedness/mirroring is the first thing to check** if the sim hand mirrors the wrong
  way: MediaPipe labels left/right from the camera's view, `to_l20_side` assumes a
  convention, and a selfie-recorded clip is already mirrored. Confirm this before
  suspecting the retargeter.
- Optionally commit a short sample clip to fixtures so the demo is reproducible without
  the user supplying a file; otherwise take the path as an argument.

## Context to load (nothing more)
root + `src/perception/CLAUDE.md`, `src/perception/` (HandSource + `MediaPipeHandSource`
+ pipeline + frame), `src/viz/` (the existing loop), `STATE.md`, this amendment.
