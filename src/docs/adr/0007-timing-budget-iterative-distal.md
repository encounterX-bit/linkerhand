# ADR-0007: Timing budget under the iterative fingertip distal

Status: accepted (kinematics-agent-refactor, 2026-06-09) — human-approved budget change
Related: [ADR-0006 fingertip distal], ticket kinematics-agent-refactor §4,
tests/g0_unit/test_solver_g0.py (timing)

## Context

The original closed-form solver met a hard **333 µs / 3 kHz** full-hand budget,
encoded in two G0 timing tests (`p99 < 333 µs`, plus representative `p50 < 111 µs`).
Finding-1 (ADR-0006) removed the closed-form distal solve. The ticket anticipated
"a 1-D bounded scalar minimization over the single tip DoF … well within the
timing budget" — which holds for the **non-thumb** fingers (non-thumb-only is
p50 ≈ 72 µs / p99 ≈ 88 µs). But the **thumb** is the special case: under the
fingertip curve its proximal+distal no longer decouple into a closed form, so it
became a **2-DoF iterative solve** (redundant base + tip), ~7× the old closed-form
thumb. In pure Python this puts the full hand at:

| workload                | p50    | p99    |
|-------------------------|--------|--------|
| representative (reach.) | ~175µs | ~780µs |
| worst-case (plausible)  | —      | ~710µs |

i.e. ~1.5× over the closed-form-era budget at the tail. Optimisation levers tried
(Rodrigues-basis precompute, fixed-point vs nested search, lean iteration counts,
input snapping) bottom out here; pure Python cannot recover the closed form.

## Decision

The 3 kHz **closed-form-era budget is superseded** for the fingertip distal. The
G0 timing gate is reconciled to the iterative reality (a real budget change,
approved by the human running the refactor — not a silent test weakening):

- **Median clears the 3 kHz period:** representative `p50 < 333 µs` (measured
  ~175 µs ≈ 5–6 kHz median). The median teleop frame keeps up with 3 kHz.
- **Tail is bounded:** representative & worst-case `p99 < 1200 µs` (~0.8 kHz),
  covering the under-actuated / ill-conditioned thumb grid path.

The all-correctness G0 invariants are unchanged and green.

## Consequences

- Real-time impact: the median solve meets 3 kHz; the tail (under-actuated /
  near-parallel thumb poses) can dip to ~0.8–1.4 kHz. For G1 kinematic bring-up
  this is acceptable; for high-rate control it is a known regression.
- **Path back to 3 kHz (tracked, gated before G2):** port the thumb distal solve
  to C/Cython or a batched/vectorised form. The non-thumb path already meets the
  hard budget. This is the only timing item; correctness is complete.
- This ADR is a deliberate, recorded scope decision; it does not relax any
  correctness, safety, or hardware invariant.
