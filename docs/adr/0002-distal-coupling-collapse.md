# ADR-0002: Distal coupling — human PIP+DIP collapse to one L20 "tip" DoF

Status: accepted (G0; confirmed against the URDF in Step 0)
Related: [ADR-0001], [ADR-0003], hardware/LIMITS.md, ticket §2/§3

## Context

The human finger has two independently flexing distal segments (PIP, DIP); the
thumb has one (IP). The L20 commands a single "tip" value per finger. Step 0
inspection of `linker-bot/linkerhand-urdf` (vendored at `src/sim/urdf/l20/`)
showed the URDF models the distal as **two revolute joints coupled by a fixed
`mimic` ratio**, not one joint:

- non-thumb: `*_dip = 0.8917 · *_pip` (driver = `*_pip`, the tip command)
- thumb:     `thumb_ip/dip = 1.1619 · thumb_mcp` (driver = `thumb_mcp`, the tip
             command)

So the 21 URDF revolute joints reduce to **16 independent DoF = the 16 actuated
L20 DoF**, and the coupling makes "human PIP and DIP collapse to one aligned
direction" physically exact on this hand.

## Decision

Collapse the human distal segments to a single aggregate distal direction and
align it with the L20's single tip DoF:

- Human: `u_dist = unit(L_d - L_b)` (the c landmark — DIP/IP — is not used; ADR-0003).
- Robot: `r_dist = unit(P_c - P_b)`, the *middle* phalanx bone (non-thumb) /
  *proximal* phalanx bone (thumb). Because the mimicked DIP/IP is a fixed-ratio
  slave and the finger flexion axes are parallel, this bone rotates **purely about
  the single tip axis** as the tip command varies → the distal alignment is an
  exact 1-DoF Paden–Kahan subproblem-1 (ADR-0004). The slaved DIP/IP follows by
  the URDF ratio and is **not** aligned separately.

The forward-kinematics ORACLE (`eval/reference_solver/`, yourdfpy) applies the
mimic ratios automatically; the closed-form solver bakes the equivalent
zero-pose tip axis and never needs the ratio explicitly (it only commands the
driver joint, which is what the L20 SDK expects).

## Consequences

- The L20 fingertip over-curls relative to the human aggregate (the slaved
  DIP/IP adds flexion). This is a fidelity question deferred to G1; for G0
  orientation alignment the single-bone representative is the correct, exact
  model of the under-actuated distal DoF.
- comms/SDK must command only the driver joints (idx 15–19); never command the
  mimic joints. Driver ranges in hardware/LIMITS.md already keep the slaved
  joints in range (1.57·0.8917=1.40; 1.05·1.1619=1.22).
