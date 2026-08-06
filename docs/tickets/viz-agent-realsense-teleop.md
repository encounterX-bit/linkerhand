# Ticket: `viz-agent` — Live RealSense Teleop Visualizer (post-G2, sim-only)

**Owns:** new `src/viz/`; **authorized to add ONE new backend file** in
`src/perception/` (`realsense_source.py`, a new `HandSource`) without modifying
existing perception code.
**Gate:** none — this is a visualization layer on top of G2, not a new gate. It runs
entirely in sim: **no `HW_ENABLE_TOKEN`, no `src/comms`, no actuation.**
**Depends on (all read-only / additive):** `src/perception/` (HandSource interface,
palm-plane frame transform, one-euro smoothing, pipeline), `src/kinematics/` (FK +
conventions + mimic), `finger_retarget.retarget()`, `safety.filter()`, `src/sim/`
(PyBullet + URDF load), the URDF, the committed `synthetic_openclose` sequences.
**Runtime deps (documented, not in CI):** `pyrealsense2`, `mediapipe`, `opencv`.
**Done =** Stage 1 win condition met + the headless equivalence test green. Stage 2 is
exploratory (report, don't gate).

> Additive only. Do not modify existing perception/retarget/filter/sim/kinematics —
> import them read-only. The one new perception file follows the existing
> `HandSource` interface. Keep smoothing in perception, never in the solver path.

---

## Goal
See it work: your hand in front of the RealSense, the sim L20 mirroring it in a
PyBullet window, driven through the *real* retarget + safety filter — then attempt
in-hand rotation of a supported object. Two stages, the second gated behind the first.

## STAGE 1 — live RealSense → sim hand mirror (the milestone)

### 1a. `src/perception/realsense_source.py` (new `HandSource`)
- Capture aligned RGB + depth (`pyrealsense2`, `rs.align(rs.stream.color)`); pull
  color-stream intrinsics.
- MediaPipe Hands on the RGB → 21 2D landmarks. For each, sample the **aligned depth
  map** at its pixel and deproject (`rs2_deproject_pixel_to_point`) → metric 3D in the
  camera frame. **Use the RealSense depth map, not MediaPipe's estimated world-z** —
  measured metric depth is the whole point of this backend.
- Depth holes (common at fingertips/edges): small-neighborhood median, else hold-last
  / flag low confidence (reuse the perception pipeline's robustness path).
- Set the depth-confidence signal **HIGH** (metric) vs the RGB backend's LOW.
- Emit metric 3D camera-frame points into the existing pipeline. **The palm-plane
  frame transform is metric-agnostic and rigid-recovering, so it converts these to the
  hand_base frame unchanged** — the backend's only job is good metric 3D; downstream is
  untouched.

### 1b. `src/viz/` live GUI loop
- PyBullet **GUI** (`p.connect(p.GUI)`), load the L20 URDF.
- Real-time loop at camera rate (~30 Hz): `hand_landmarks` → `retarget()` →
  `safety.filter()` → set the 16 joints **kinematically** (`resetJointState`) +
  enforce mimics from `src/kinematics` (crispest for a mirror — no PD lag, no contact;
  dynamics is Stage 2). Handedness via perception's `to_l20_side`. Smoothing on.
- Optional: a cv2 window showing the camera feed + MediaPipe overlay beside the sim.
- **Camera-free fallback flag:** replay the committed `synthetic_openclose` sequence
  through the identical loop, so it runs and is testable with no RealSense attached.

**Win condition:** your hand moves, the sim hand follows, and it looks right —
**including the thumb** (the thumb tracking cleanly is the tell that the depth wiring
is doing its job).

## STAGE 2 — cradled object, attempt in-hand rotation (exploratory)
- Switch the loop to **dynamics** (reuse `src/sim` G2 dynamics/closed-loop).
- Add a **supported** object — palm-backed or in a shallow cradle, **never
  free-floating** (per G2 finding #2: free objects eject with no wrist to pre-load).
  Slow, ramped contact.
- Drive with live vision (or a recorded rotation gesture). Observe how far rotation gets.
- **This is exploration, not pass/fail.** Report what works and the failure modes
  (ejection, slippage, thumb behavior). Expect partial/unstable rotation — it stresses
  exactly the weak points: monocular-vs-depth thumb fidelity, distal under-actuation,
  and the **still-open thumb-palm collision blind spot** (rotation drives the thumb
  across the palm). Stage-2 results are **provisional until that filter fix lands** —
  note it; nothing here touches hardware so it's fine to proceed.

## Tests
- **Deprojection (no camera needed):** mock intrinsics + synthetic depth → assert
  pixel+depth deprojects to the expected 3D point; depth-hole handling returns
  finite/flagged, never NaN.
- **Pipeline equivalence (the key automated check):** replaying `synthetic_openclose`
  through the viz loop produces the **same joint trajectory** as the existing
  G1/G2 pipeline on the same input (proves the viz loop is wired correctly, not a new
  code path that drifts). Run headless.
- **Contract conformance:** the RealSense backend emits valid `hand_landmarks`,
  confidence HIGH.
- Stage 1 visual win condition and Stage 2 rotation are **manual** observations — no
  automated gate.

## Notes
- Stage 1 first as its own deliverable. Isolating variables matters: get the empty-hand
  mirror crisp before adding contact, so when the object misbehaves in Stage 2 the only
  new variable is the physics.
- Do not advance to hardware. Do not modify the safety filter to "fix" the thumb-palm
  gap here — that's a separate safety ticket; just note where it bites.
- On finish: update `STATE.md` with the Stage 1 result, the depth-confidence behavior
  observed live, and the Stage 2 findings.

## Context to load (nothing more)
root + `src/perception/CLAUDE.md`, `src/perception/` (HandSource + pipeline + frame),
`src/kinematics/`, `finger_retarget.retarget()`, `safety.filter()`, `src/sim/`,
the URDF, the `synthetic_openclose` fixtures, this ticket.
