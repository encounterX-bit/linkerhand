#!/usr/bin/env bash
# Bootstrap the SEW-style Linker Hand L20 retargeting repo for Claude Code.
# Usage:  mkdir l20-retarget && cd l20-retarget && bash /path/to/bootstrap.sh
set -euo pipefail

mkdir -p src/{perception,finger_retarget,safety,comms,sim}
mkdir -p tests/{g0_unit/fixtures,g1_kinematic,g2_dynamic,g3_hardware}
mkdir -p eval/{benchmarks,reference_solver}
mkdir -p docs/{adr,tickets} contracts hardware

# ---------------------------------------------------------------- root CLAUDE.md
cat > CLAUDE.md <<'EOF'
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
EOF

# -------------------------------------------------------------------- STATE.md
cat > STATE.md <<'EOF'
# STATE

Active gate: G0 (solver unit tests, CPU only)

## Next
1. eval-agent: build eval/reference_solver/ (slow scipy oracle, see ticket §6).
2. solver-agent: implement src/finger_retarget/ until tests/g0_unit/ is green.

## Blocked
- hardware/LIMITS.md needs real joint ranges from linker-bot/linkerhand-urdf.

## Handoff log
- (bootstrap) repo scaffold created.
EOF

# ------------------------------------------------------------------- contracts
cat > contracts/hand_landmarks.schema.json <<'EOF'
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "hand_landmarks",
  "type": "object",
  "required": ["side", "landmarks", "frame", "t"],
  "properties": {
    "side":  { "enum": ["left", "right"] },
    "frame": { "const": "hand_base" },
    "t":     { "type": "number" },
    "landmarks": {
      "type": "array", "minItems": 21, "maxItems": 21,
      "items": { "type": "array", "minItems": 3, "maxItems": 3,
                 "items": { "type": "number" } },
      "description": "MediaPipe Hands 21-landmark convention, 3D, hand_base frame"
    }
  }
}
EOF

cat > contracts/l20_targets.schema.json <<'EOF'
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "l20_targets",
  "type": "object",
  "required": ["side", "joint_rad", "active_idx", "clamped", "t"],
  "properties": {
    "side":    { "enum": ["left", "right"] },
    "clamped": { "const": true },
    "t":       { "type": "number" },
    "joint_rad": {
      "type": "array", "minItems": 20, "maxItems": 20,
      "items": { "type": "number" },
      "description": "20 entries; idx 11-14 MUST be 0.0 (reserved)"
    },
    "active_idx": {
      "type": "array",
      "const": [0,1,2,3,4,5,6,7,8,9,10,15,16,17,18,19]
    }
  }
}
EOF

# ------------------------------------------------------------- module stubs
for m in perception finger_retarget safety comms sim; do
cat > "src/$m/CLAUDE.md" <<EOF
# CLAUDE.md — src/$m

See repo-root CLAUDE.md for invariants. Write ONLY in this module.
Coordinate with other modules via contracts/*.schema.json only.
Task spec: docs/tickets/  (the ticket naming this module).
Done = the gate's named tests pass. Update STATE.md on finish.
EOF
done

# --------------------------------------------------------------- LIMITS.md
cat > hardware/LIMITS.md <<'EOF'
# hardware/LIMITS.md  (FILL BEFORE G1)

Source: linker-bot/linkerhand-urdf (clone and read joint limits).

| idx | name             | min_rad | max_rad |
|-----|------------------|---------|---------|
| 0   | thumb base       | TODO    | TODO    |
| ... | ...              | ...     | ...     |
| 19  | little tip       | TODO    | TODO    |

## Hardware bring-up clamps (G3)
- FORCE / CURRENT cap: TODO  (start FAR below 100 N grip max)
- speed cap: TODO
- e-stop: TODO (physical + software)
EOF

cat > README.md <<'EOF'
# l20-retarget
SEW-style per-finger retargeting onto the Linker Hand L20. See CLAUDE.md and
docs/ARCHITECTURE.md. Build order and gates in docs/gates.md / STATE.md.
EOF

echo ".venv/"        >  .gitignore
echo "__pycache__/" >>  .gitignore

git init -q 2>/dev/null || true
git add -A 2>/dev/null || true
git commit -qm "scaffold: SEW-style L20 hand retargeting" 2>/dev/null || true

cat <<'DONE'

Scaffold created. Next:
  1. Save the 3 provided docs into the repo:
       docs/ARCHITECTURE.md   <- SEW-L20-HandRetarget-Scaffold.md
       docs/tickets/solver-agent-G0.md  <- ticket-solver-agent-G0.md
       (keep the first overview doc as docs/ARCHITECTURE-context.md if you like)
  2. Fill hardware/LIMITS.md from linker-bot/linkerhand-urdf.
  3. Open Claude Code at this repo root and paste the kickoff prompt.
DONE
