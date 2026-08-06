# ADR-0010: Closed-form audit of the whole solver; fixed-cost analytic distal

Status: accepted (solver-agent, closed-form-restore ticket, 2026-06-16)
Related: [ADR-0002 distal-collapse], [ADR-0003 segment convention],
[ADR-0004 reduced-finger optimality], [ADR-0006 fingertip distal solve],
[ADR-0007 timing], ticket `docs/tickets/solver-closed-form-restore.md`

## Context

The ticket asked: audit every subproblem the retargeting solver runs, classify
each (closed-form / fixed-cost analytic / search), and **restore closed form**
where reachable — non-thumb distal via Paden–Kahan subproblem-1, thumb distal via
a `t = tan(θ/2)` polynomial reduction (degree ≤ 4 → analytic roots). Hard
constraint: match the fingertip oracle to the current worst-case tolerance, with
no change to the target (ADR-0003 fingertip), the mimic coupling, or any tolerance.

## Audit matrix (Phase 1)

| subproblem | DoF | before | geometric reason | after |
|---|---|---|---|---|
| non-thumb proximal | 2 | closed (PK-2) + bounded nearest | 2R cone | **unchanged** — exact subproblem-2; fixed-cost analytic nearest only when unreachable |
| non-thumb distal | 1 | grid + Brent (fixed-cost search) | epitrochoid (parallel axes, **non-integer** mimic rate) | **fixed-cost analytic** — subproblem-1 seed + capped Newton |
| thumb proximal | 3 | closed-form per `cd` (PK-2 + PK-1) | 3 CMC, 1-DoF redundancy | **unchanged** — closed-form ktip two-plane + subproblem-2/1 |
| thumb distal (+ redundant base) | 1 (+1) | cd fixed point + robust grid | coupled redundancy + same epitrochoid | **kept** — cd fixed point; inner tip-align now fixed-cost analytic |

**Proximal solves confirmed genuinely closed-form (no silent fallback crept in).**
Over 3000 random reachable configs/side the proximal residual worst-case is
**2.1e-8 rad** (machine-epsilon) for every finger — i.e. exact Paden–Kahan
subproblem-2 on reachable targets. The non-thumb base `_two_axis_base` returns the
exact subproblem-2 angles when reachable (early `<1e-9` return) and only falls to a
**bounded, fixed-cost** coordinate-descent nearest (4 subproblem-1 steps, each
closed-form) when the 2R cone misses. The thumb proximal is solved in closed form
per distal latitude `cd` (the two-plane `ktip` construction + subproblem-2 +
subproblem-1). No search is used in either proximal.

## Why the distal cannot be closed-form (the headline finding)

Both distal subproblems share one geometry. With the base fixed, the fingertip
vector from `P_b` is (ADR-0006)

    v(θ) = R(k, θ)·mvec0 + R(k, (1+ratio)·θ)·dvec0

— the middle bone `mvec0 = P_c − P_b` rotating at rate 1 and the distal bone
`dvec0 = fingertip0 − P_c` rotating at rate `1+ratio` about the **common** flexion
axis `k` (DIP/IP ∥ PIP/MCP, verified dot = 1.0 on all 10 finger/sides).

The ticket's "easy win" — parallel axes compose into one rotation by `θ(1+ratio)`
→ circle → subproblem-1 — is true only for a **single** vector rotated by *both*
joints (the distal bone alone). The fingertip is the **sum of two bones at
different rates**, so it does not compose into one rotation. Verified numerically:
`unit(v(θ))·k` (the latitude) is **not constant** — it varies by up to ~8e-4 in
cosine over a finger's tip range, with an offset up to ~0.016 (left thumb). A
curve of non-constant latitude is **not a circle about any axis**, so there is no
exact Paden–Kahan subproblem-1.

**Polynomial reduction (`t = tan(θ/2)`) — degree analysis.** The alignment
condition reduces to

    |UM|·sin(a + θ) + |UD|·sin(b + (1+ratio)·θ) = 0.

Under `t = tan(θ/2)`, `sin(a+θ)` is a rational function of `t` of degree 2, but
`sin(b + (1+ratio)θ)` is rational in `t` **only if `1+ratio` is an integer**. The
mimic rate is **non-integer**: `ratio = 0.8917` (non-thumb) and `1.1619` (thumb),
so `1+ratio = 1.8917 / 2.1619`. Nearest small fractions are `33/37` and `43/37`
(error ~2e-4). To force a polynomial one must rationalise `ratio ≈ p/q` and
substitute `ψ = θ/q`, yielding `sin(p·ψ)`-type terms — a polynomial of degree
**O(p+q) ≈ 70–140**, *and only approximate* (it would change the mimic ratio,
which the ticket forbids). So the polynomial path is **intractable** (degree ≫ 4)
and would violate "do not simplify the mimic coupling." The condition is
genuinely transcendental.

The geometry is messy because the target (the physical fingertip, ADR-0003
Finding-1) is correct. Per the ticket, we keep the correct fixed-cost method and
report why — we do not ship a wrong formula.

## Decision (Phase 2)

**Non-thumb distal → fixed-cost ANALYTIC (the ticket's sanctioned form).** Replace
the 5-point grid + 7-step Brent minimiser with:

1. a **Paden–Kahan subproblem-1 analytic seed** — rotate `v(0)` about `k` toward
   `u_dist` (the rate-1 circle the ticket suggested; exact to first order), then
2. a **hard-capped Newton** (4 steps) on `g(θ) = −dot(unit v(θ), u_dist)` using
   the *exact* `v, v′, v″` (Rodrigues basis precomputed once), which corrects the
   epitrochoid wobble — Newton converges quadratically from the seed, and
3. an **endpoint guard**: take the best of `{Newton point, lo, hi}` so a
   clamped/under-actuated optimum sitting at a tip bound is never missed.

Deterministic, bounded, no grid — *not a search*. This is exactly the ticket's
"hard-capped Newton from an analytic seed" fallback, and it realises the ticket's
subproblem-1 suggestion *as the seed*, with the Newton step being the proof that
the curve is not a circle.

**Thumb distal → kept** (the redundant base solve of ADR-0006 is correct and
fixed-cost). Its inner 1-DoF tip-align now calls the same fixed-cost analytic
solve. The outer redundancy (cmc_pitch) is closed-form per `cd` with a short
scalar `cd` fixed point and a robust grid only on the ill-conditioned /
under-actuated tail — unchanged in behaviour.

**Proximal → untouched.**

## Verification (oracle-match, both hands, ≥3000 configs/side)

Worst-case segment residual vs the fingertip target, 3000 random reachable
configs/side — **identical before and after** (no regression):

| finger | proximal worst | non-thumb distal worst | thumb distal worst |
|---|---|---|---|
| non-thumb | 2.1e-8 | 2.1e-8 | — |
| thumb | 2.1e-8 | — | 8.3e-7 (R) / 1.08e-6 (L) |

The new analytic non-thumb distal matches the prior grid+Brent minimiser to
**~4e-11 over 20,000 random directions/finger (incl. unreachable), never worse**.
G0 26/26 green. Timing improved: representative p50 220→200 µs, worst-case
p99 884→800 µs (correctness was the gate, not speed).

## Should-not-change set — all confirmed

- Oracle unchanged (`eval/` untouched); G0 26/26; proximal numbers untouched
  (2.1e-8); reachable round-trip ~1e-7 (distal 2.1e-8 / thumb 1e-6); reserved idx
  11–14 = 0; the ADR-0003 target, the mimic ratios, and all tolerances unchanged;
  hardware untouched (no `src/comms`, no `HW_ENABLE_TOKEN`). Only
  `src/finger_retarget/solver.py` changed.

## Consequences

- The fingertip distal align is now **fixed-cost analytic** for all five fingers
  (no 1-D search anywhere). The only remaining iteration is the thumb's scalar
  `cd` fixed point + tail grid (the redundant-base coupling, ADR-0006), which is
  fixed-cost and unchanged.
- ADR-0007's timing reconciliation still holds (the thumb redundant-base tail is
  the cost driver, not the distal align); the median comfortably clears 3 kHz.
- No closed-form formula was forced anywhere it would be wrong: the non-integer
  mimic ratio makes the distal transcendental, and that is recorded here.
