# ADR-0008: G2 self-collision model, projection, and filter-latency baseline

Status: proposed (safety-agent G2, 2026-06-09) — **HUMAN REVIEW REQUIRED before
merge** (root CLAUDE.md: changes to src/safety need explicit human review)
Related: ticket docs/tickets/safety-agent-G2-v2.md; [ADR-0005 FK authority],
[ADR-0007 timing]; src/safety/, tests/g2_safety/

## Context

G2 needs an inline guard `candidate l20_targets -> SAFE l20_targets` that is a
*projection* (returns the nearest collision-free, in-limits, rate-bounded
config), not a checker. It runs on hardware later, so it must be **sim-free**
(no PyBullet, no runtime mesh load) and use the single `src/kinematics` FK
authority — no third FK. Decisions that needed recording:

1. **What the collision proxy is**, given the URDF ships only STL collision
   meshes and the joint limits already bound finger flexion against the palm.
2. **How the projection maps a Cartesian non-penetration correction back to
   joint space** without a second FK.
3. **What the timing gate is**, now that the loop runs at 30 Hz camera rate.

## Decision

### Collision proxy (baked offline, mesh-free at runtime)
- **One capsule per phalanx** = the rigid phalanx link as a segment between two
  FK link origins (distal link: origin→physical fingertip), radius = **half the
  smallest collision-mesh bounding-box extent** (the cross-section). Half the
  *smallest* extent (not the mean of the two minor extents) keeps adjacent
  fingers — spaced ~0.022 m at rest — collision-free with a small separation
  margin. Radii are baked in `collision_model.CAPSULE_RADII`; regenerate with
  `_gen_collision_model.py` (trimesh, dev-only) if the meshes change.
- **Palm = a palmar HALF-PLANE + y/z footprint, NOT a box.** A box over the
  base-link AABB false-positives on a *natural fist*: the four fingertips
  legitimately come to rest just outside the palmar skin (measured: outside the
  actual palm mesh by 4–10 mm; min fist-tip x ≈ 0.013 m). The half-plane
  (outward normal +x, `x0 = -0.005`) only flags a tip driven *through* the
  palmar surface into the palm body — on this hand essentially a thumb-into-palm
  event, since finger flexion is limit-bounded away from the palm. Validated:
  rest and full-fist poses are collision-free under the model.

### Projection (XPBD-style, fixed iteration, one FK per iteration)
Self-collision is resolved by a **fixed-iteration position-based projection**
over the three ticket pair categories (adjacent fingers, thumb-vs-finger,
fingertip-vs-palm). Each iteration calls FK **once**; the Cartesian
non-penetration correction is mapped to joint space by the **analytic
rigid-body Jacobian** read off the same FK link transforms (joint world axis =
`R_child · axis_local`, origin = child-link origin; column = `axis × (contact −
origin)`), scaled per-constraint by `depth / |∇sep|²` (a damped-least-squares /
XPBD-compliance-0 step). The mimic DIP/IP contribution to a distal-phalanx
contact is included automatically by walking parent joints (mimic attributed to
its driver idx × ratio). This is **not** a second FK — it differentiates the FK
the authority already computed. Fixed count → deterministic + real-time.

**Guard composition (chatter-free):** the candidate is first projected inside
the **full-limits** box to a *rate-independent* stable target, then rate-limited
toward that target within a band around `prev_safe`, then (only if the rate
limit actually clipped) re-projected inside the band so the output is always
non-penetrating mid-approach. Projecting before rate-limiting is what removes
boundary chatter — the target a held candidate converges to no longer depends on
`prev_safe`. Static guards (limits incl. mimic-tightened tip ranges, reserved
idx 11–14 = 0, non-finite sanitisation) bound the box, so the output satisfies
every guard regardless of projection convergence.

### Timing gate (two-part)
The loop runs at 30 Hz, so the whole retarget→filter→command budget is one
frame = **33,333 µs** (ADR-0007 retired the 3 kHz budget). The filter gate is:
- **(a) Absolute:** every call (incl. the deep-collision worst case) < 33,333 µs.
  Measured worst ≈ 11.9 ms (~3× headroom; solver tail ~0.78 ms leaves the rest).
- **(b) Regression:** filter p99 over a fixed representative+adversarial workload
  ≤ committed baseline × (1 + margin). Committed baseline **11,500 µs**
  (measured best-of-3 p99: right ~11.4 ms, left ~11.9 ms; p50 ~2.3 ms;
  collision-free fast path ~0.85 ms). The tail is the deep-collision third of
  the workload running the full PBD budget.

## Open items for human sign-off
- **`FILTER_LATENCY_REGRESSION_MARGIN` (proposed 0.50).** Wide enough to absorb
  shared-CI jitter, tight enough to trip a real algorithmic slowdown (e.g.
  doubling iterations). Baseline is machine-specific; re-measure on the target
  machine and adjust if it moves materially.
- **Force-clamp & watchdog specs** (`config.ForceClampSpec` / `WatchdogSpec`):
  15 N grip cap («100 N), 0.6 A current, 0.20 s stale timeout → open-hand safe
  pose. These are **specs only**; `comms` enforces at G3 and a HUMAN sets
  `HW_ENABLE_TOKEN`. No actuation in this module.

## Consequences
- True non-penetration is guaranteed only w.r.t. this capsule+plane proxy, not
  the full meshes; the margin absorbs proxy slack. Tighten radii/margin if the
  proxy proves optimistic against mesh contact in sim.
- Do NOT advance past G2 on the agent's own (human gate).
