# ADR-0004: Optimality of the closed-form solver for the reduced finger

Status: accepted (G0)
Related: [ADR-0001], [ADR-0002], [ADR-0003], ticket §4

## Context

The ticket requires the optimality claim to be **re-derived for the reduced
finger**, not inherited from the paper's 7-DoF arm result. Objective per finger:

    J = w_prox · ∠(r_prox, u_prox) + w_dist · ∠(r_dist, u_dist)   (geodesic)

## Decision & proof sketch

**Distal (1-DoF), all fingers — globally optimal.**
With the base fixed, `r_dist(φ) = rot(k_tip, φ) · r_dist_ref` traces a circle on
the unit sphere (a single rotation about the fixed tip axis `k_tip`; valid because
the mimicked DIP/IP is a fixed-ratio slave about a parallel axis — ADR-0002).
Minimising `∠(r_dist(φ), u_dist)` is the projection of `u_dist` onto that circle:
the minimiser is `φ* = atan2(k·(a×b), a·b)` with `a,b` the components of
`r_dist_ref, u_dist` perpendicular to `k_tip` (Paden–Kahan subproblem-1). This is
the exact global minimiser of the distal term; if `φ*` violates a joint limit the
objective is monotonic in `φ` on each side of the optimum, so the clamped value is
the constrained optimum. ⇒ distal term is globally optimal (reachable or not).

**Base, non-thumb (2-DoF) — exact when reachable, nearest otherwise.**
`r_prox = rot(k_abd,θ_a) rot(k_base,θ_b) r_prox0`. When `u_prox` lies in the 2R
image, Paden–Kahan subproblem-2 returns the exact angles ⇒ prox term = 0 (global
min). When outside (joint limits / cone), we return the nearest reachable
orientation (bounded closed-form coordinate steps + clamping). The base axes do
not affect the distal axis materially for non-thumb fingers, so the two terms are
solved independently and both reach their constrained optima ⇒ J is minimised.

**Thumb (4-DoF, 3 of which align a 2-DoF proximal direction) — exact when
reachable.**
The 3 CMC axes give a 1-DoF redundancy. Crucially the redundancy couples the two
terms: the CMC config rotates the tip axis, so it must serve the distal term too.
Because the tip axis equals the palm-normal `s` (`thumb_mcp` ∥ `cmc_yaw`/`cmc_pitch`),
s-component invariance gives two linear constraints on the unit tip axis `ktip`:
`u_prox·ktip = r_prox0·s` and `u_dist·ktip = r_dist0·s`. Their intersection with
the unit sphere yields `ktip` in closed form (≤2 solutions); `(opp,abd)` then follow
from subproblem-2 and `(base,tip)` from subproblem-1 — aligning BOTH segments
exactly (J = 0) whenever a real `ktip` exists and the angles are in range. When no
real `ktip` exists or limits bind, a bounded base sweep returns the nearest pose.

**Scope.** We do NOT claim the paper's 7-DoF reachability/optimality. We claim:
(i) the distal term is globally optimal always; (ii) the per-finger J reaches 0
for reachable targets (verified: ~3e-6 over 2000 round-trip poses/side); (iii)
otherwise a nearest-reachable pose with `J_cf ≤ J_oracle + ε` (verified vs the
scipy oracle). Weights `w_prox, w_dist` are configurable (default 1/1).
