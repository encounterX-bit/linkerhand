# ADR-0001: Per-finger SEW-style orientation alignment (not full-hand IK)

Status: accepted (G0)
Related: docs/ARCHITECTURE.md §1, ticket solver-agent-G0

## Context

We retarget human hand landmarks to the Linker Hand L20. The L20 is heavily
under-actuated per finger (3 DoF non-thumb, 4 DoF thumb) with a single coupled
distal "tip" command and no fingertip roll. The SEW-Mimic method
(arXiv:2602.01632) aligns limb segment *orientations* via closed-form geometric
subproblems for a redundant 7-DoF arm.

## Decision

Apply SEW-Mimic's transferable core **per finger**, treating each finger as a
mini-chain:

- Define human segment unit vectors between landmarks (orientation, not position).
- Solve **closed-form** for joint angles that align the robot phalanx directions
  to the human directions, via Paden–Kahan subproblems.
- Use **orientation error**, not Euclidean — so the mapping is calibration/scale
  free across human/robot size (no per-user calibration).

Two decoupled subproblems per finger: a BASE alignment (proximal direction) and a
DISTAL alignment (single tip DoF). See ADR-0003 for the exact segment convention
and ADR-0002 for the distal coupling.

**Dropped from the paper** (do not port): the wrist-orientation stage (no
fingertip roll DoF), and the SEW-angle / elbow-swivel redundancy
parameterisation (fingers are under-actuated, not redundant). The thumb is the
one finger with internal redundancy (3 CMC axes aligning a 2-DoF direction);
that 1-DoF redundancy is resolved by the distal objective, not by a swivel angle.

## Consequences

- Per-finger solves are independent and embarrassingly parallel; the full hand is
  16 angles from 21 landmarks with no global optimisation.
- "Optimality" must be redefined for the reduced finger (see ADR-0004): minimum
  total per-segment orientation error, NOT the paper's 7-DoF pose-reach result.
