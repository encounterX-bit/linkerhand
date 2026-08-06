# Task: fetch + stream example hand videos through the sim

**Status:** `VideoHandSource` and the `--source video` loop option already exist. This
adds no module logic — it obtains footage and runs it.
**Owns:** an `assets/` or `tests/fixtures/video/` addition + a small runner/report; no
changes to `VideoHandSource`, the loop, or the pipeline.
**Gate:** none — sim-only, hardware-free.
**Runtime deps:** `mediapipe`, `opencv` (already present); network access on this machine.
**Done =** ≥1 clip with good detection streams through the loop into the GUI, with
per-clip detection rates reported and capture artifacts saved.

---

## What this is (and isn't)
This exercises the **video → MediaPipe front end** plus the plumbing, on real footage.
It's monocular RGB — weak z — so it does **not** validate depth/retargeting accuracy
(that's the RealSense path, already built). Soft thumb / mushy depth-axis motion is the
expected monocular limit, not a failure.

## Steps
1. **Fetch 2–3 openly-licensed clips.** Use clearly free-to-use sources only
   (Pexels, Pixabay, Mixkit, Wikimedia Commons, or MediaPipe's example assets) —
   **not** scraped/YouTube content. Pick clips with a **single, prominent, well-lit
   hand, palm-ish toward camera, clear finger motion** (open/close, some rotation),
   5–15 s. Record source URL + license for each in a short `NOTICE`/README.
2. **Validate detection BEFORE streaming.** Run MediaPipe over each clip and report the
   **per-frame hand-detection rate** and which hand it sees. Discard clips that detect
   poorly (e.g. <70% of frames). This is the step that separates "bad clip" from
   "broken pipeline" — do not skip it, and do not stream a clip you haven't validated.
3. **Stream the survivors** through the existing loop
   (`--source video --video-path <clip>`) into the PyBullet GUI. Save a short capture
   (gif/mp4 or frame dump) per clip to the viz `out/` dir for review.
4. **Handedness check** per clip: MediaPipe labels left/right from the camera's view and
   stock clips may be either hand or mirrored — confirm `to_l20_side` maps correctly;
   note any clip that needed a mirror flip.
5. **Report** (STATE.md + the saved artifacts): per-clip detection rate + which hand,
   which clips tracked cleanly, observed thumb/depth softness, any mirroring fixes.

## Guardrails
- **Licensing:** free-to-use only; log source + license per clip. Do not commit large
  files — keep at most one tiny clip in fixtures for reproducibility, gitignore the
  rest, and record provenance.
- **If this machine can't reach the video sites,** say so and fall back to the
  recorded-sequence replay path (file-free) rather than fabricating a result.
- Success must be **evidenced** (real detection numbers + saved capture artifacts), not
  asserted.

## Note
A stranger's hand at an awkward angle tracks worse than a well-framed one, so a poor
clip is not a pipeline verdict. The cleanest sanity check remains your own footage or
the recorded-sequence replay — the fetched clips are for "watch it move from real
video," not for accuracy.

## Context to load
root, `src/perception/video_source.py`, `src/viz/` (the loop + its source options),
`STATE.md`, this task.
