# Ticket: `safety-agent` — Self-Collision Filter + Command Guards (G2, updated)

**Module (write only):** `src/safety/` (+ its tests)
**Gate:** G2 (built and validated in sim; force-clamp spec also feeds G3)
**Depends on:** `contracts/l20_targets.schema.json`; **FK + conventions from
`src/kinematics/`** (now the single authority — limits, mimic ratios, joint/segment
maps all live there); L20 URDF **collision** geometry in `src/sim/urdf/` (read-only);
`hardware/LIMITS.md`.
**Human review required before merge** (safety module, per root CLAUDE.md).
**Done =** `tests/g2_safety/` green + filter exposed as a callable the G2 harness inserts.

---

## Goal
The inline guard between the retargeter and the commanded config:
`candidate l20_targets → SAFE l20_targets`. A **projection**, not a checker — returns
the nearest collision-free, in-limits, rate-bounded config. Single-hand rescope of
SEW-Mimic's XPBD self-collision filter (inter-finger + finger-palm). Runs on hardware
later, so **no PyBullet, no sim dependency** in this module.

## FK source (resolved — do not re-open)
`src/kinematics/` is the single FK authority. **Import FK and conventions from it.
Do NOT fork a third FK.** PyBullet is sim-only and must not appear here.

## Interface (the seam with sim-agent — lock this first)
```
filter(candidate: l20_targets, prev_safe: l20_targets, dt: float)
    -> { joint_rad[20], clamped: true, modified: bool, reason: str|null }
```
Pure, deterministic, sim-free. `prev_safe` enables rate limiting and continuity.

## Responsibilities
1. **Self-collision projection (XPBD-style).** Build a lightweight collision model
   from the URDF collision meshes — capsule per phalanx + palm slab — and run a
   fixed-iteration position-based non-penetration solve over pairs: adjacent fingers,
   thumb vs each finger, fingertips vs palm. Configurable separation margin. Fixed
   iterations → real-time + deterministic. Use `src/kinematics` FK for link poses.
2. **Static guards (last line of defense):** clamp to `hardware/LIMITS.md` even
   though the solver already clamps; enforce idx 11–14 == 0; enforce mimic
   consistency on the driver joints (ratios from `src/kinematics`).
3. **Rate limiting.** Bound per-joint velocity (config rad/s) vs `prev_safe` so a
   perception glitch / teleport can't command a jump.
4. **Force-clamp spec.** Grip-force / motor-current cap as config — start FAR below
   100 N. This module *specifies*; `comms` enforces at G3. No actuation here.
5. **Watchdog spec.** Stale-input timeout → safe hold/open pose. Spec only.

## Tests (`tests/g2_safety/`)
1. **Idempotence.** Already-safe configs pass through unchanged within ε (`modified==false`).
2. **Projection correctness.** Interpenetrating configs (crossed fingers, thumb into
   palm, over-closed fist) → collision-free output, minimally changed from candidate.
3. **Adversarial set (paper ablation analog).** Self-collision scenarios → no
   penetration after filtering.
4. **Rate limiting.** Input teleport → output respects max joint velocity given `dt`.
5. **Continuity / no chatter.** Sweep safe → colliding → safe; output continuous,
   no boundary oscillation.
6. **Limits / reserved.** Output in-range; idx 11–14 == 0; no NaN.
7. **Determinism + timing (two-part gate).** Fixed iterations, deterministic.
   (a) *Absolute:* solver tail + filter + overhead must fit `LOOP_PERIOD = 33333 µs`
   (one 30 Hz camera frame) — trivially true, but it's the real-time guarantee.
   (b) *Regression:* filter p99 latency must not exceed its committed baseline by more
   than `FILTER_LATENCY_REGRESSION_MARGIN`. Record the filter p99 and commit it as the
   baseline so silent slowdowns get caught long before they'd threaten the period.
8. **Config sanity.** Force/current cap present and ≤ a defined fraction of max;
   watchdog safe-pose defined and in-limits.

## Timing note (resolved: 30 Hz camera → `LOOP_PERIOD = 33333 µs`)
The 3 kHz budget is retired. The whole retarget → filter → command loop runs at
camera rate (30 Hz), so the budget for the entire loop is one frame = 33,333 µs.
The solver thumb is now iterative (p50 ~175 µs full hand, tail ~780 µs) — about 2.3%
of the period, ~40× headroom — so the filter has the rest of the frame to work in;
iteration count is the lever only in the unlikely event it ever runs tight. The
filter's gate is therefore the two-part test above: a hard 33,333 µs ceiling (real-time
guarantee) plus a regression guard against its own committed baseline (catches silent
slowdowns that would still be under the ceiling). C/Cython on the thumb is NOT needed.

## Context to load (nothing more)
root + `src/safety/CLAUDE.md`, `contracts/l20_targets.schema.json`, `src/kinematics/`
(FK + conventions), the URDF collision geometry, `hardware/LIMITS.md`, this ticket.
