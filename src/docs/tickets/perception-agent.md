# Ticket: `perception-agent` — Vision → `hand_landmarks`

**Module (write only):** `src/perception/` (+ its tests)
**Gate:** feeds G0/G1 (runs in parallel with `sim-agent`)
**Depends on:** `contracts/hand_landmarks.schema.json`, ADR-0003 (frame convention).
**Done =** perception tests green + emits valid `hand_landmarks` + a recorder that
dumps replayable sequences.

---

## Goal
Camera → 21 3D landmarks in the **hand_base frame defined by ADR-0003**, emitting
the `hand_landmarks` contract at camera rate, smoothed, and **estimator-agnostic**.

## Implementation
1. **Backend.** MediaPipe Hands (RGB) as default. Wrap behind a `HandSource`
   interface so an RGB-D variant or a MANO estimator (HaMeR/WiLoR) can drop in later
   without changing the output contract.
2. **Frame transform.** Map raw estimator output into the hand_base frame **exactly**
   as ADR-0003 defines (origin at wrist landmark 0; axes from the hand). This must
   match the solver/oracle frame or everything is silently rotated — the
   perception-side analog of the thumb-label trap. Add a frame-convention test
   against a known pose.
3. **Handedness.** Map MediaPipe's camera-view left/right labeling to the L20 side
   convention used on `/cb_*_hand_*` topics. Test both sides.
4. **Smoothing.** One-euro (or equivalent) filter on the landmarks, configurable,
   **on by default** — monocular z is jittery. Filter the INPUT here; keep
   `finger_retarget` a pure function. Do NOT put smoothing in the solver hot path.
5. **Depth.** Default to estimator-native z; structure so an aligned RGB-D source can
   replace z. Log a depth-confidence signal so weak-z frames are visible downstream.
6. **Recorder.** Dump landmark sequences (conforming to `hand_landmarks`) to
   `tests/g1_kinematic/fixtures/real/`. **This is the convergence point with
   `sim-agent`** — its real-set residual test replays these.
7. **Robustness.** Handle no-detection / low-confidence frames (hold last good + flag),
   never emit garbage or NaN.

## Tests
- Schema conformance: 21 pts, hand_base frame, all finite.
- Frame-convention: known pose → expected landmark geometry (guards the silent-rotation trap).
- Smoothing reduces jitter on a noisy synthetic stream (variance drop) within a lag bound.
- Handedness mapping correct, both sides.
- Rate/latency meets camera rate; no-detection handled gracefully.
- Runs on recorded video / synthetic streams (no live camera needed for CI). Note
  real-camera validation — where depth quality is judged — as a manual step.

## Context to load (nothing more)
root + `src/perception/CLAUDE.md`, `contracts/hand_landmarks.schema.json`,
ADR-0003, this ticket.
