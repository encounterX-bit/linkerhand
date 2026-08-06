# CLAUDE.md — ROOT INVARIANTS (read first, every task)

## What we build
One pipeline: human hand landmarks -> 16 Linker Hand L20 joint radians, via a
per-finger SEW-style orientation-alignment solver (from arXiv:2602.01632).
No arm. The L20 is the whole embodiment. Input is camera/vision-based.

## L20 facts (do not re-derive)
- 16 actuated DoF: idx 0-4 base(flex), 5-9 abduction, 10 thumb opposition,
  15-19 tip(distal). idx 11-14 are RESERVED -> always command 0.0.
- Per finger 3 DoF (base+abduction+tip); thumb 4 (adds opposition).
- "tip" is ONE coupled distal command: human PIP and DIP collapse to one aligned
  direction. No fingertip roll, no redundancy/swivel.
- Command in RADIANS over CAN topic /cb_*_hand_control_cmd_arc; state on
  /cb_*_hand_state_arc. URDF: linker-bot/linkerhand-urdf. Sim: PyBullet/MuJoCo.

## Method scope
KEEP: vector-between-keypoints, closed-form subproblem alignment, orientation
(not Euclidean) error. DROP: wrist-alignment stage, SEW-angle redundancy.
Objective = min total per-finger segment-orientation error.

## Hard safety laws
- NEVER actuate real hardware. src/comms drivers MUST refuse unless env
  HW_ENABLE_TOKEN is set, and only a HUMAN sets it.
- NEVER advance past gate G2 on your own. G3+ is human-gated.
- Force/current MUST be clamped far below 100 N during bring-up.
- Respect joint ranges in hardware/LIMITS.md. Changes to src/safety need
  explicit human review.

## How to work
- Read your module's CLAUDE.md + contracts/*.schema.json before coding.
- Write ONLY in your assigned module. Coordinate via contracts only.
- Done = the named tests pass. Append a handoff note to STATE.md each turn.
- Record decisions as docs/adr/ files; never silently reverse one.

## Current state -> see STATE.md
