# Ticket: `solver-agent` — Per-Finger SEW Retargeting Solver (Gate G0)

**Module (only write here):** `src/finger_retarget/`
**Gate:** G0 (CPU only — no hand, no sim)
**Depends on contracts:** `contracts/hand_landmarks.schema.json` (in),
`contracts/l20_targets.schema.json` (out)
**Definition of done:** all of `tests/g0_unit/` green + timing budget met.

---

## 1. Goal

Implement a closed-form, per-finger orientation-alignment solver that maps 21 3D
hand landmarks (MediaPipe convention) to 16 actuated Linker Hand L20 joint angles
(radians), reserved joints (idx 11–14) = 0. Method = SEW-Mimic applied per finger
(see ARCHITECTURE.md §1). Estimator-agnostic: consumes the landmark contract, not
a specific camera.

## 2. Inputs / outputs

```
in:  hand_landmarks {21 × [x,y,z] in hand_base frame, side}
out: l20_targets    {joint_rad[20] (idx 11-14 = 0), active_idx, clamped=True}
```

Per-finger landmark groups (MediaPipe indices):
- thumb: 1(CMC) 2(MCP) 3(IP) 4(TIP)
- index: 5 6 7 8 · middle: 9 10 11 12 · ring: 13 14 15 16 · little: 17 18 19 20
- wrist: 0

Human segment unit vectors per finger:
- `u_prox` = normalize(PIP − MCP)
- `u_dist` = normalize(TIP − PIP)   ← aggregate distal (PIP+DIP collapsed; see ADR-0002)

## 3. Algorithm (per finger)

Each L20 finger is a reduced chain. Solve two decoupled subproblems.

**(A) Base alignment — set the proximal-phalanx direction.**
Non-thumb: 2 DoF {base (flex), abduction (spread)}. Thumb: 3 DoF {base, abduction,
opposition}. Find joint angles that rotate the L20 proximal-phalanx unit vector
`r_prox(θ)` to align with `u_prox`. This is "align one vector to a target by
successive rotations about known joint axes":
- Closed-form via Paden–Kahan subproblem-2 (two-axis) for fingers; three-axis for
  the thumb (subproblem-2 then a residual subproblem-1 about the opposition axis).
- If `u_prox` is outside the reachable cone (joint limits), return the **nearest
  reachable** orientation (clamp each angle to its URDF limit after solving).

**(B) Distal alignment — set the distal direction.**
1 DoF {tip}, a rotation about the finger flexion axis `k_tip`. Find the tip angle
that best aligns `r_dist(φ)` with `u_dist`:
- Closed-form via Paden–Kahan subproblem-1: the angle-minimizer is the rotation
  that brings `r_dist` as close as possible to `u_dist` in the plane ⟂ `k_tip`
  (project both onto that plane, take the signed angle, clamp to tip limits).

**Reserved joints (11–14):** always 0.

## 4. Objective & optimality (re-derive for the reduced finger)

Minimize total geodesic orientation error:
```
J(finger) = w_prox · ∠(r_prox, u_prox) + w_dist · ∠(r_dist, u_dist)
```
- The 1-DoF distal solve is a projection onto a circle ⇒ provably the angle
  minimizer (document the proof in `docs/adr/` or a solver docstring).
- The 2-DoF (thumb 3-DoF) base solve is exact when `u_prox` is reachable, else the
  nearest point on the reachable manifold.
- Do **not** claim the paper's 7-DoF optimality result; it does not transfer.
- `w_prox, w_dist` configurable per finger (default 1.0/1.0).

## 5. Constraints / guards

- All outputs clamped to URDF joint ranges (load limits from `src/sim/` URDF or a
  cached `hardware/LIMITS.md`). Never emit out-of-range or NaN.
- Pure function, no I/O, no hardware import. Deterministic.
- No optimizer in the hot path — closed-form only. (The optimizer lives only in
  `eval/reference_solver/`.)

---

## 6. G0 test suite (`tests/g0_unit/`)

**Oracle (`eval/reference_solver/`):** slow `scipy.optimize.minimize` over each
finger's actuated DoF minimizing the §4 objective via the URDF forward kinematics.
Ground truth for accuracy tests. (Built first, by `eval-agent`.)

Tests:
1. **Matches oracle.** N≥1000 random plausible hand poses: closed-form objective
   `J_cf ≤ J_oracle + ε` (ε small, e.g. 1e-3 rad). Closed-form should match or beat
   the local optimizer.
2. **Per-segment error bound.** For reachable targets, `∠(r,u) ≤ tol` after solve.
3. **Joint limits.** Every output within URDF range; reserved idx 11–14 == 0 exactly.
4. **Scale invariance (calibration-free).** Multiply all landmarks by k∈{0.5,2,5}
   ⇒ identical joint output (orientation-only).
5. **Degenerate poses.** Fully extended (collinear segments), fully curled,
   abduction extremes, target near a joint axis ⇒ finite, clamped, no NaN.
6. **Determinism.** Same input → bitwise-identical output across runs.
7. **Handedness.** Left/right landmark sets map to the correct side's joint signs.
8. **Timing benchmark.** Per-finger and full-hand (5 fingers) solve time on the
   target CPU; report p50/p99. Budget: full hand ≪ 333 µs (3 kHz). Fail if p99 > budget.

Fixtures: a small set of recorded/synthetic landmark frames (flat hand, fist,
pinch, point, thumbs-up) committed under `tests/g0_unit/fixtures/`.

---

## 7. Agent context (what to load, nothing more)

- `CLAUDE.md` (root) · `src/finger_retarget/CLAUDE.md`
- `contracts/hand_landmarks.schema.json` · `contracts/l20_targets.schema.json`
- this ticket · the failing tests in `tests/g0_unit/`
Do **not** read other `src/` modules. Coordinate only via the two contracts.
On finish: append to `STATE.md` (what changed / next / blocked); record the
distal-collapse and optimality decisions as ADRs if not already present.

## 8. Assumption to confirm

Written against MediaPipe's 21-landmark 3D convention. If perception ends up using
a MANO-based estimator (HaMeR/WiLoR), the only change is a keypoint-extraction
adapter in `src/perception/` that emits the same 21 landmarks — this ticket is
unaffected.
```
