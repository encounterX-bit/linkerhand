# ADR-0003: Per-finger segment-direction convention (oracle + solver share)

Status: accepted (G0, established by eval-agent in Step 1);
        AMENDED 2026-06-09 by kinematics-agent (Finding-1 endpoint fix, see below)
Related: [ADR-0002 distal-collapse], [ADR-0005 FK authority], [ADR-0006 fingertip
         endpoint], ticket solver-agent-G0 §2–§4, ticket kinematics-agent-refactor

> **AMENDMENT (Finding-1, 2026-06-09).** The robot distal endpoint `P_c` below is
> superseded: `r_dist` now runs to the **physical fingertip** (the tip of the last
> distal link, *including* the DIP/IP mimic curl), not to the DIP/IP joint origin.
> See the "Finding-1 endpoint update" section at the end and ADR-0006. Everything
> else in this ADR (human side, `r_prox`, objective, scale-freedom) is unchanged.

## Context

The oracle (`eval/reference_solver/`) and the closed-form solver
(`src/finger_retarget/`) live in isolated modules and coordinate only through the
two contracts. But both must score the SAME orientation-error objective `J`, or
the G0 "closed-form matches oracle" test is meaningless. The objective needs a
precise, shared definition of the human target directions (`u_prox`, `u_dist`)
and the robot segment directions (`r_prox`, `r_dist`). This ADR fixes that
convention. The solver re-implements it from baked constants; it does not import
the oracle.

## Decision

**Human side — positional landmark groups.** Each finger owns four MediaPipe
landmark indices `[a, b, c, d]`:

| finger | a | b | c | d |
|--------|---|---|---|---|
| thumb  | 1 (CMC) | 2 (MCP) | 3 (IP)  | 4 (TIP) |
| index  | 5 | 6 | 7 | 8 |
| middle | 9 | 10| 11| 12|
| ring   | 13| 14| 15| 16|
| little | 17| 18| 19| 20|

    u_prox = unit(L_b - L_a)
    u_dist = unit(L_d - L_b)      # aggregate distal; the c landmark is collapsed

For non-thumb fingers this is the ticket's `u_prox = PIP-MCP`, `u_dist = TIP-PIP`.
For the thumb (which has no PIP) the same positional rule makes `u_prox` the
*metacarpal* bone (CMC→MCP) and `u_dist` the MCP→TIP aggregate — which is exactly
what the robot's thumb DoF control (see below).

**Robot side — joint-origin bones.** From URDF forward kinematics in the hand
base frame, take three joint-origin points per finger (`P_a`, `P_b`, `P_c`) and

    r_prox = unit(P_b - P_a)      # set by the BASE DoF
    r_dist = unit(P_c - P_b)      # set by the single TIP DoF

| finger | P_a (link origin) | P_b | P_c | base DoF | tip DoF |
|--------|-------------------|-----|-----|----------|---------|
| thumb  | thumb_metacarpals_base1 | thumb_proximal | thumb_distal | 3 CMC (idx 0,5,10) | thumb_mcp (idx 15) |
| non-thumb | *_proximal | *_middle | *_distal | mcp_pitch + mcp_roll (base,abd) | *_pip (idx 16–19) |

`r_dist` is the *middle*-phalanx bone (non-thumb) / *proximal*-phalanx bone
(thumb). Because the DIP/IP joint is a fixed-ratio URDF `mimic` of the tip driver
and the finger flexion axes are parallel, this bone rotates **purely about the
single tip axis** as the tip command varies → the distal solve is an exact 1-DoF
Paden–Kahan subproblem-1 (the angle minimiser). See ADR-0002.

**Objective.** `J = w_prox·∠(r_prox,u_prox) + w_dist·∠(r_dist,u_dist)`, geodesic
angles, default weights 1.0/1.0 (`eval/reference_solver/objective.py`).

## Consequences

- Mesh-free: only joint origins are needed, so no fingertip/STL length estimate
  enters the convention. (The true fingertip would make `u_dist` depend on bone
  lengths and break the clean single-axis distal rotation.)
- Calibration/scale-free: all four vectors are unit, so the mapping is invariant
  to the scale of the landmark cloud (G0 test 4).
- The robot's *actual* fingertip over-curls relative to the human aggregate
  because the mimicked DIP/IP adds flexion. That fidelity question is deferred to
  G1; for G0 alignment the middle/proximal bone is the correct, exact
  representative of the under-actuated distal DoF.
  *(Finding-1 amendment: this deferral is now resolved — see below. The fingertip
  IS the endpoint; the "over-curl" is exactly the signal we now align to.)*

## Finding-1 endpoint update (2026-06-09, kinematics-agent)

**Decision.** The robot distal direction is

    r_dist = unit(fingertip - P_b)

where `fingertip` is the tip of the last distal link — `link_c`'s frame applied
to a body-fixed local tip offset (`conventions.TIP_LOCAL`, the distal-mesh vertex
farthest from that link's own DIP/IP joint origin). Because the distal link frame
carries the DIP/IP `mimic` curl, this fingertip moves with the coupled distal
joint. This replaces the previous `r_dist = unit(P_c - P_b)` (the DIP/IP joint
origin), which was mimic-*independent* and so could never see the curl.

**Why.** The human side is unchanged: `u_dist = unit(L_d - L_b) = TIP − PIP`,
which spans BOTH human distal segments. The old robot `r_dist` (a single inter-
joint bone) was the wrong analogue — it ignored the distal phalanx. The fingertip
endpoint makes `r_dist` the true PIP→fingertip vector, matching `u_dist` segment-
for-segment. (Empirically the two robot definitions differ by ~10° mean, up to
~20°: exactly the distal curl the old convention dropped.)

**Geometry — still a single tip DoF, but a curve not a circle.** The DIP/IP axis
is exactly parallel to the PIP/MCP tip-driver axis (verified, dot = 1.0), and
DIP/IP = `ratio · tip` (0.8917 non-thumb, 1.1619 thumb). So with the base config
fixed, the fingertip vector is the exact closed form

    v(θ) = L_m · R(k, θ) · m0  +  L_d · R(k, (1+ratio)·θ) · d0
    r_dist(θ) = unit(v(θ))

with `k` the tip axis, `m0 = unit(P_c−P_b)`, `L_m = |P_c−P_b|`,
`d0 = unit(fingertip0−P_c)`, `L_d = |fingertip0−P_c|` (all baked at the zero pose,
then rotated by the base solve). This is two parallel-axis rotations at rates `1`
and `1+ratio`, so `r_dist(θ)` traces a curve (an epicycle), NOT a single circle —
the distal alignment is therefore **no longer an exact Paden–Kahan subproblem-1**.
It is now a 1-DoF bounded scalar minimisation of `∠(r_dist(θ), u_dist)` over the
tip limit (the solver re-derivation; ADR-0006). `r_prox` and the base solve are
untouched and remain exact.

**Consequence for "mesh-free".** The original mesh-free property is relaxed: one
mesh-derived constant (the local tip offset) now enters, baked once into
`conventions.TIP_LOCAL` so neither the runtime FK nor the solver hot path loads a
mesh. Scale/translation-freedom on the HUMAN side is unaffected (still unit
vectors); the robot side is fixed metric geometry.
