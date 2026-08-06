# ADR-0006: Fingertip distal target + its (non-closed-form) solve

Status: accepted (kinematics-agent-refactor, 2026-06-09)
Related: [ADR-0002 distal-collapse], [ADR-0003 segment convention],
[ADR-0004 reduced-finger optimality], [ADR-0007 timing], ticket
kinematics-agent-refactor

## Context

ADR-0003 originally defined the robot distal direction as `r_dist = unit(P_c - P_b)`
— the bone to the DIP/IP joint ORIGIN, which is mimic-*independent* and so never
saw the coupled distal curl. The human side `u_dist = TIP − PIP` spans BOTH human
distal segments, so this was the wrong analogue (Finding-1). The human confirmed
`r_dist` must run to the PHYSICAL fingertip, including the DIP/IP mimic curl.

## Decision

**Target.** `r_dist = unit(fingertip - P_b)`, where `fingertip` is the last distal
link's frame applied to a body-fixed local tip offset (`conventions.TIP_LOCAL`,
the distal-mesh vertex farthest from the DIP/IP joint). The distal link frame
carries the mimic curl, so the fingertip moves with the coupled distal joint. See
the ADR-0003 Finding-1 amendment.

**Geometry — a curve, not a circle.** The DIP/IP axis is exactly parallel to the
tip-driver axis (verified, dot = 1.0) and DIP/IP = `ratio · tip`. With the base
config fixed, the fingertip vector is the exact closed form

    v(θ) = R(k, θ)·mvec0 + R(k, (1+ratio)·θ)·dvec0

(`mvec0 = P_c − P_b`, `dvec0 = fingertip0 − P_c`, baked then base-rotated; verified
vs FK to ~1e-9). Two parallel-axis rotations at rates `1` and `1+ratio` make a
curve (an epicycle), so `r_dist(θ)` is **NOT** a single rotation of a fixed vector
— **there is no Paden–Kahan subproblem-1 closed form** for the distal align.

**Solve.** The proximal/base solve is UNCHANGED and exact (subproblem-2). The
distal solve is the ticket-sanctioned numerical fallback:

- *Non-thumb:* a bounded 1-D minimisation of `∠(v(θ), u_dist)` over the tip limit
  (safeguarded-parabolic Brent on `−cos∠`, which is smooth at the optimum;
  Rodrigues basis precomputed). Cheap — well within the timing budget.
- *Thumb:* the old two-plane closed form (ADR-0004) is **invalid** here: it
  required `u_dist·ktip` to be a constant latitude, but for the fingertip curve
  `v·ktip` is constant while `|v|` varies, so `r_dist·ktip = vk/|v(θ)|` is not.
  The redundant cmc_pitch + tip are therefore solved **jointly** by a fixed point
  on the distal latitude `cd = vk/|v(θ*)|` (the base is closed-form per `cd`, so
  no grid in the common case), with a robust base grid for the near-parallel /
  ill-conditioned cases (`u_prox ∥ u_dist`) and the under-actuated nearest. Any
  result with a non-small total residual is routed to the grid, which keeps the
  solve robust on ALL reachable targets (worst over 3000 random configs/side =
  0; G0 cache reachable worst 6.4e-7).

**Scale invariance.** The iterative distal can flip a discrete branch under a
1-ULP change in `u_prox/u_dist` (e.g. landmarks scaled by a non-power-of-2). The
unit target directions are snapped to ~1e-12 rad so all scales give bit-identical
solver input — restoring exact scale invariance (the old closed form was scale-
exact algebraically). The snap is ~5 orders below the solve precision.

## Consequences

- ADR-0004's exact thumb two-plane is superseded for the distal; the proximal
  subproblem-2 it described is retained.
- Oracle distal ground truth regenerated (plausible J_oracle rose — the under-
  actuated distal is honestly farther from the fingertip-inclusive target).
- G1 distal residual re-measured (real set: overall p95 0.197 → 0.178; the curl
  is now scored). The reachable round-trip stays exact (~1e-6).
- The thumb solve is iterative, so the closed-form-era 3 kHz timing budget no
  longer holds at the tail — see ADR-0007.
