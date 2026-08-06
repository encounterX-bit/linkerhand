# Ticket: Finger-Contact Safety Deep Dive (investigate → propose)

**Scope:** **read-only on `src/safety/`** — this ticket changes NO pipeline code. It
produces an investigation report, committed gesture fixtures, and a fix *proposal*.
The actual safety change is a separate, human-reviewed ticket.
**Gate:** none — analysis. Sim-only, hardware-free.
**Depends on (read-only):** `src/perception/` (landmarks + pipeline), `src/kinematics/`
(FK + conventions + collision-relevant geometry), `finger_retarget.retarget()`,
`src/safety/` (filter + collision model, run but not modified), `src/sim/` + viz, the
processed video sequences. Dev dep: `python-fcl`/`trimesh` for full-mesh ground truth
(measurement only — no repo change, as in the conservativeness review).
**Done =** Phase-1 report with per-gesture localization backed by artifacts +
committed OK / fingers-crossed fixtures + Phase-2 written proposal & draft ADR.
`src/safety/` untouched.

---

## Principle
The filter isn't malfunctioning on these gestures — it's a uniform-margin self-collision
*avoidance* projection, and the OK sign and crossed fingers are gestures whose point is
finger *contact*. The filter can't distinguish **intended contact** (fingertip surfaces
touching) from **harmful penetration** (a finger body driven through another). The deep
dive's job is to localize where each gesture actually breaks before proposing any change.

## PHASE 1 — investigate & localize (decompose the pipeline per gesture)
Use the actual failing inputs. Identify the OK-sign and fingers-crossed sequences that
exhibit the failure (from the processed MOVs / recorded fixtures); **commit them as named
fixtures** (`tests/fixtures/gestures/{ok_sign,fingers_crossed}.json` or the video) so the
investigation is reproducible and the eventual fix has a regression target.

For **each** gesture, produce an artifact at every stage and a localization verdict:

1. **Perception (check this first — it's the sneaky one).** Overlay the detected
   landmarks on the source frames. The OK sign occludes thumb/index tips; a crossed
   finger hides behind its neighbor — MediaPipe may misplace them, in which case the
   retargeter is faithfully mapping garbage and no downstream fix helps. Report whether
   landmarks are trustworthy on these frames.
2. **Retarget output vs full mesh.** Run `retarget()` on the (good) landmarks → the
   **unfiltered** config. Run a full-mesh FCL collision check on it (reuse the
   conservativeness-review setup). Split the cases: meshes only *touch/approach* →
   filter is over-conservative; meshes *deeply penetrate* → geometry/kinematic.
3. **Margin sweep.** Sweep `separation_margin_m` ∈ {0, 0.5, 1, 2 mm} and report at what
   value the OK sign closes / the fingers approach. Isolates the margin as culprit.
4. **Kinematic reachability (esp. fingers-crossed).** Search abduction/config space:
   does *any* config cross the fingers (or close the OK loop) **without** mesh
   penetration? If none exists, it's a hardware kinematic limit (L20 fingers move in
   near-parallel planes with limited abduction), not a filter bug.
5. **Sim contact realism.** With the filter allowing contact, observe PyBullet: do
   fingertips interpenetrate, bounce apart, or rest? Render unfiltered-vs-filtered
   side-by-side for both gestures.

**Phase-1 output:** for each gesture, a verdict — perception / retarget / filter
over-conservatism / kinematic limit / sim-contact — with the artifacts that prove it.
Likely (to be confirmed): OK sign = filter over-conservatism + sim-contact realism;
fingers-crossed = possible kinematic limit.

## PHASE 2 — propose (no implementation)
Based only on what Phase 1 shows, write the proposal + a draft ADR:
- **Contact-aware filter:** an *allowed-contact set* — fingertip-fingertip pairs (pinch,
  OK, tripod) allowed to contact, projected only against deep penetration; finger-body↔
  body stays forbidden with a real margin. The principled "not all contact is unsafe."
- **Sim fingertip compliance:** contact stiffness/damping on the fingertip shapes so
  contact *rests* softly instead of clipping or popping (realism for the OK sign).
- **Collision-geometry fidelity:** fingertip spheres + properly fitted link capsules
  instead of the ½-smallest-extent capsules. **Reconcile this with the still-open
  thumb-palm / conservativeness item** — make it one coherent collision-model rework,
  not two conflicting ones.
- **Graceful degradation:** for any genuinely unreachable gesture (likely crossing),
  output the closest non-penetrating approximation rather than refusing or forcing it.

## Guardrails
- **Do NOT modify `src/safety/`** (or retarget/kinematics/perception). Run them, measure
  them, propose against them. Filter changes are a separate reviewed ticket.
- Findings must be **evidenced** (overlays, FCL verdicts, sweep numbers, renders), not
  asserted. Keep the FCL/measurement scripts but don't wire them into the pipeline.
- Commit the gesture fixtures; keep large media gitignored with provenance.

## Context to load
`STATE.md`, `CLAUDE.md`, `src/perception/`, `src/kinematics/`, `src/safety/` (read),
`finger_retarget`, `src/viz/`, ADR-0008 (collision model), the conservativeness-review
note, this ticket.
