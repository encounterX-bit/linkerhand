# Ticket: `sim-agent` — G2 Dynamic / Contact Closed-Loop Sim (updated)

**Module (write only):** `src/sim/` (+ `tests/g1_kinematic/`, `tests/g2_dynamic/`)
**Gate:** G2 (dynamics + contact — last sim gate before hardware)
**Depends on:** the G1 harness; full L20 URDF (masses/inertias); **FK + conventions
from `src/kinematics/`** (PyBullet = dynamics only); `safety-agent`'s `filter()`;
recorded sequences in `tests/g1_kinematic/fixtures/real/`;
`finger_retarget.retarget()` (read-only); ADR-0003; `hardware/LIMITS.md`.
**Done =** Step 0 makes G1 an honest 56/56, then `tests/g2_dynamic/` green + a
closed-loop grasp demo artifact.

---

## STEP 0 — make G1 genuinely green (do this first)
G1 is currently 55/56: the single fail-closed `test_real_residual` lumps proximal
(exact) with distal (under-actuation-limited). Now that the metric is trustworthy
(Finding 1 fixed, `r_dist` fingertip-inclusive), split it:

1. **`test_proximal_residual` — HARD GATE.** Proximal-segment geodesic error on the
   recorded real set; assert p95 ≤ `PROXIMAL_TOL`, a tight bound you justify from the
   observed near-zero values (real-set prox p50 = 0.000). Real pass/fail. This is what
   makes G1 honestly green and satisfies the entry gate below.
2. **`test_distal_residual_monitored` — MONITOR + REGRESSION GUARD.** Compute distal
   error (overall + per-finger p50/p95), write to `tests/g1_kinematic/out/` and the
   handoff. Pass by default — do NOT enforce an absolute quality line on *synthetic*
   data. But **hard-fail on regression** beyond `REGRESSION_MARGIN` above the committed
   baseline (post-refactor: overall p95 ≈ 0.178, thumb dist ≈ 0.143). When
   `G1_DISTAL_RESIDUAL_THRESHOLD` is later set from real-camera data, it also becomes an
   absolute gate.
3. Commit the baseline distal numbers so the regression guard has a reference.

Result: G1 = 56/56 honest green — proximal hard-gated, distal monitored.

## ENTRY GATE for the dynamic work (both hold after Step 0)
- **Finding 1 resolved** ✓ (kinematics refactor — `r_dist` is fingertip-inclusive).
- **G1 green with proximal hard-gated** — produced by Step 0.

## Goal
Promote the G1 harness to a dynamic, closed-loop contact sim and prove the full
teleop loop is stable and safe under dynamics + latency + the safety filter.

## Build
1. **Dynamics.** L20 with masses/inertias; gravity + contact. `setJointMotorControl2`
   (position/PD) instead of `resetJointState`. Enforce the 5 mimic joints **under
   dynamics** (gear constraint or per-step; verify ratios hold while stepping — harder
   than kinematic). Document PD gains; note they're sim-only, NOT hardware gains.
2. **Closed loop.** landmarks → `retarget()` → `safety.filter()` → command → step →
   read → repeat. Inject CAN/control latency as a delay buffer. Run at the real
   control rate.
3. **Grasp / contact.** Small object set: a cylinder (power grasp) + a small sphere
   (pinch). Pinch is included on purpose — it stresses the fingertip geometry the
   Finding-1 fix corrected.
4. **Force cap (virtual).** Simulated grip/contact force ≤ the safety-defined cap.
   Real clamp is comms/G3.

Use `src/kinematics/` for the **metric** FK; PyBullet provides dynamics only. Import
`retarget()` and `safety.filter()` read-only.

## Tests (`tests/g2_dynamic/`)
1. **Loop rate (two-part gate, `LOOP_PERIOD = 33333 µs` = one 30 Hz frame).** The
   whole retarget → filter → command loop runs at camera rate. Measure p99 *compute*
   time = solver (incl. iterative thumb tail) + filter + sim step; the latency buffer
   is modeled delay, NOT compute, so it does not count against the period.
   (a) *Absolute:* p99 compute must be < 33,333 µs (real-time guarantee; expect ~1 ms,
   so this passes with large margin).
   (b) *Regression:* p99 compute must not exceed the committed baseline by more than
   `LOOP_REGRESSION_MARGIN` — catches a silent 10× slowdown that would still be under
   the ceiling. Commit the measured baseline.
2. **Mimic under dynamics.** Coupling ratios (0.8917 / 1.1619) hold within tolerance
   while stepping.
3. **Grasp.** Hand closes on each object; stable contact (not ejected/exploded);
   force ≤ virtual cap.
4. **Filter ablation (paper analog).** Adversarial self-collision scripts: filter ON
   → no penetration; filter OFF → penetration. Proves the filter is load-bearing.
5. **Latency stability.** At latencies {low, target, high}: loop stable, bounded
   tracking error, no divergence.
6. **Limits / reserved / no-NaN** under dynamics.
7. **Tracking penalty.** Per-segment error on recorded sequences with dynamics+latency
   vs the G1 kinematic numbers — quantify and report (monitored, not a hard gate).

## Notes
- Build dynamics / mimic-under-dynamics / grasp / latency tests FIRST — they don't
  need the filter. Leave the ablation + closed-loop-with-filter tests until safety's
  `filter()` lands (the sync point).
- Keep the object set tiny and deterministic (fixed seeds/poses).
- If grasp goes unstable or the loop diverges under latency, report it as a tuning
  finding (PD gains / filter iterations) — do NOT force a green.
- On finish: update `STATE.md`. If anything in Step 0's should-be-unchanged set moves
  (G0, round-trip, proximal), stop and report.

## Context to load (nothing more)
root + `src/sim/CLAUDE.md`, the two contracts + the safety `filter()` interface,
the full URDF, `src/kinematics/` (FK + conventions), recorded sequences, ADR-0003,
`hardware/LIMITS.md`, this ticket.
