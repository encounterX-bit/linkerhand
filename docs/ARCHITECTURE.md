# SEW-Style Hand Retargeting → Linker Hand L20 — Agentic Scaffold (v2, hand-only)

Single-pipeline scaffold for retargeting human hand motion to the **Linker Hand
L20** using the orientation-alignment method from SEW-Mimic (`arXiv:2602.01632`),
applied **per finger**. No arm. Driven by Claude Code with scoped agents and
sim-before-hardware gates.

> Supersedes the earlier two-pipeline (arm) draft. There is one pipeline:
> human hand landmarks → 16 L20 joint radians.

---

## 0. Hardware facts (the scaffold is built around these)

**Source repos**
- ROS SDK: `github.com/linker-bot/linkerhand-ros-sdk` (commands in radians over CAN)
- URDF: `github.com/linker-bot/linkerhand-urdf`
- Sim: official PyBullet + MuJoCo assets (PyBullet: `rosrun linker_hand_pybullet linker_hand_pybullet.py _hand_type:=L20`)

**Control interface**
- Position control in **joint-space radians** over CAN @ 1 Mbit/s.
- Command topic: `/cb_left_hand_control_cmd_arc` (and `_right_`).
- State topic: `/cb_left_hand_state_arc` (radians + angular range).
- Pressure/tactile sensors available (`TOUCH: True`).

**L20 joint map — 20 named entries, 16 actuated, 4 reserved**
```
idx  name              role                     SEW analog
0    Thumb base        MCP flexion              "shoulder" axis A
1    Index base        MCP flexion              "shoulder" axis A
2    Middle base       MCP flexion              "shoulder" axis A
3    Ring base         MCP flexion              "shoulder" axis A
4    Little base       MCP flexion              "shoulder" axis A
5    Thumb abduction   MCP spread               "shoulder" axis B
6    Index abduction   MCP spread               "shoulder" axis B
7    Middle abduction  MCP spread               "shoulder" axis B
8    Ring abduction    MCP spread               "shoulder" axis B
9    Little abduction  MCP spread               "shoulder" axis B
10   Thumb opposition  CMC opposition           thumb "shoulder" axis C
11   Reserved          —                        unused
12   Reserved          —                        unused
13   Reserved          —                        unused
14   Reserved          —                        unused
15   Thumb tip         distal flexion (coupled) "elbow"
16   Index tip         distal flexion (coupled) "elbow"
17   Middle tip        distal flexion (coupled) "elbow"
18   Ring tip          distal flexion (coupled) "elbow"
19   Little tip        distal flexion (coupled) "elbow"
```
**Key kinematic facts that shape the solver**
- Per non-thumb finger: **3 DoF** = base (flex) + abduction (spread) + tip (distal).
- Thumb: **4 DoF** = base + abduction + opposition + tip.
- **PIP and DIP are NOT independent** — one "tip" command drives the whole distal
  flexion as a coupled unit. The human's two distal segments collapse to one.
- No fingertip roll → **no wrist stage**. Under-actuated → **no swivel/redundancy**.

---

## 1. The SEW-style mapping (what we keep, what we drop)

Treat each finger as a mini-SEW chain. From MediaPipe Hands (21 landmarks) per
finger: MCP, PIP, DIP, TIP. Human segment vectors: proximal (MCP→PIP),
distal-aggregate (PIP→TIP).

**Keep (the transferable core of SEW-Mimic):**
- Define limb (phalanx) **unit vectors** between keypoints.
- Solve **closed-form** for joint angles that **align robot phalanx directions to
  human phalanx directions** via geometric subproblems.
- **Orientation error, not Euclidean** → calibration-free across human/L20 size.

**Per-finger solve (non-thumb):**
1. *Base alignment ("shoulder", 2 DoF):* rotate the L20 proximal-phalanx direction
   to the human MCP→PIP vector using {base, abduction}. Closed-form
   rotate-vector-about-two-axes subproblem.
2. *Distal alignment ("elbow", 1 DoF):* rotate the L20 distal link to the human
   PIP→TIP aggregate direction using {tip}. 1-DoF closed-form.

**Thumb:** add the opposition DoF as a third base axis before distal alignment.

**Drop (do NOT port from the paper):**
- Wrist-orientation alignment stage (no fingertip roll DoF).
- SEW-angle / elbow-swivel redundancy parameterization (fingers are
  under-actuated, not redundant).
- Bimanual-arm self-collision filter → **replace** with inter-finger / finger-palm
  collision (same XPBD idea, much smaller scene).

**Redefine "optimality":** with 3–4 DoF you cannot match full fingertip pose.
The objective is **minimum total orientation error over the two finger segments**
(optionally weighted). State this explicitly; the optimality proof must be
re-derived for the reduced finger, not inherited from the 7-DoF arm.

---

## 2. Single-pipeline architecture + repo layout

```
human hand landmarks ──▶ finger_retarget ──▶ safety ──▶ comms ──▶ L20
   (MediaPipe/glove)      (per-finger SEW)    (collision   (CAN/ROS
                                               + force)     radians)
                                   │
                                   └──▶ sim (PyBullet/MuJoCo, linkerhand-urdf)

repo/
  CLAUDE.md                 # ROOT INVARIANTS (read first, every task)
  ARCHITECTURE.md           # this doc
  STATE.md                  # current gate + handoff notes
  docs/
    gates.md  agents.md  interfaces.md
    adr/ 0001-sew-per-finger.md  0002-distal-coupling-collapse.md
  contracts/
    hand_landmarks.schema.json   # 21 MediaPipe pts (or glove equivalent)
    l20_targets.schema.json      # 16 active joint radians + reserved zeros
  src/
    perception/             # hand source -> normalized landmarks   (+CLAUDE.md)
    finger_retarget/        # per-finger SEW-style solver           (+CLAUDE.md)
    safety/                 # inter-finger collision + force clamp   (+CLAUDE.md)
    comms/                  # L20 CAN/ROS driver (radians)           (+CLAUDE.md)
    sim/                    # linkerhand-urdf + PyBullet/MuJoCo      (+CLAUDE.md)
  tests/
    g0_unit/  g1_kinematic/  g2_dynamic/  g3_hardware/   # g3 token-gated
  eval/
    benchmarks/             # per-finger timing (3 kHz budget), accuracy
    reference_solver/       # slow optimization oracle for cross-checks
  hardware/
    LIMITS.md               # joint ranges, FORCE/CURRENT CLAMP, e-stop
    bringup_checklist.md
```

---

## 3. Context preservation (multi-agent)

1. **State on disk, not chat.** Specs/contracts/decisions are files; chat is
   disposable. A fresh agent re-reads `CLAUDE.md` + module `CLAUDE.md` + contract.
2. **CLAUDE.md hierarchy.** Root = invariants (the mapping, gate rules, safety
   laws). Per-module = local conventions + interface + one "current/next" line.
3. **Contracts are the only coupling.** `contracts/*.schema.json` is what agents
   agree on; they never read each other's internals.
4. **Tests = done.** "Make `tests/g0_unit/` pass against this contract." Drift dies.
5. **STATE.md + ADRs.** Terse handoff note each turn; decisions recorded once.
   Parallel work uses a **git worktree/branch per agent**; orchestrator merges.

---

## 4. Agent topology

One long-lived **orchestrator**; scoped ephemeral **subagents**, one module each:

| Agent | Owns | Gate |
|---|---|---|
| `solver-agent`  | `src/finger_retarget/` per-finger SEW solver + optimality | G0 |
| `perception-agent` | `src/perception/` landmark source + normalization | G0/G1 |
| `sim-agent`     | `src/sim/` urdf load + PyBullet/MuJoCo harness | G1/G2 |
| `safety-agent`  | `src/safety/` inter-finger collision + force clamp | G2 |
| `comms-agent`   | `src/comms/` CAN/ROS driver (radians) | G3 |
| `eval-agent`    | `eval/`, `tests/` oracle, benchmarks, regression | all |

Each subagent gets: root + module `CLAUDE.md` + its contract + the failing test.
Writes only in its module.

---

## 5. Staged gates (sim → hardware; G3+ human-only)

**G0 — Solver unit (CPU, no hand/sim).** Per-finger solver matches the slow
reference oracle on N random landmark sets within orientation-error tolerance;
optimality (min segment-orientation error) holds for the reduced finger;
per-finger + full-hand timing meets the 3 kHz budget; reserved joints (11–14)
held at zero; outputs within joint ranges from URDF. → exit: `tests/g0_unit/` green.

**G1 — Kinematic sim (PyBullet/MuJoCo, no dynamics).** Recorded hand-tracking →
16 joint radians → FK in `linkerhand-urdf`; finger directions track human within
tolerance; joint limits respected; thumb opposition calibrated. → `tests/g1_kinematic/` green.

**G2 — Dynamic / contact sim.** Full loop with dynamics + injected CAN latency;
inter-finger / finger-palm collision filter catches scripted self-collisions
(grasp closures); grip simulated within a **virtual force cap**; loop holds rate.
→ `tests/g2_dynamic/` green; **HIL eligible.**

**G3 — Hardware (real L20) [HUMAN-GATED].** Entry: G2 green **AND** human commit
sets `HW_ENABLE_TOKEN`; `hardware/LIMITS.md` reviewed; hand **securely mounted**;
**force/current clamped well below the 100 N max**; e-stop + watchdog live; nothing
in the workspace. Procedure: slow, supervised, single-finger first → full hand;
human on e-stop. → exit: `hardware/bringup_checklist.md` signed by a human.

**G4 — Task / data collection.** Full-speed within validated limits; grasp/pinch
tasks; collect demos; validate retargeting smoothness for downstream policy use.

**Structural guardrail:** `src/comms/` refuses to actuate without `HW_ENABLE_TOKEN`
(human-set only). Agent CI runs sim only (G0–G2). Agents cannot self-promote to G3.

---

## 6. Root CLAUDE.md (drop in)

```markdown
# CLAUDE.md — ROOT INVARIANTS (read first, every task)

## What we build
One pipeline: human hand landmarks -> 16 Linker Hand L20 joint radians, via a
per-finger SEW-style orientation-alignment solver (from arXiv:2602.01632).
No arm. L20 is the whole embodiment.

## L20 facts (do not rederive)
- 16 actuated DoF: idx 0-4 base(flex), 5-9 abduction, 10 thumb opposition,
  15-19 tip(distal). idx 11-14 RESERVED -> always command 0.
- Per finger 3 DoF (base+abduction+tip); thumb 4 (adds opposition).
- "tip" is ONE coupled distal command: human PIP and DIP collapse to one
  aligned direction. No fingertip roll, no redundancy/swivel.
- Command in RADIANS over CAN topic /cb_*_hand_control_cmd_arc; state on
  /cb_*_hand_state_arc. URDF: linker-bot/linkerhand-urdf. Sim: PyBullet/MuJoCo.

## Method scope
KEEP: vector-between-keypoints, closed-form subproblem alignment, orientation
(not Euclidean) error. DROP: wrist-alignment stage, SEW-angle redundancy.
Objective = min total segment-orientation error per finger.

## Hard safety laws
- NEVER actuate real hardware. src/comms drivers must refuse unless env
  HW_ENABLE_TOKEN is set, and only a HUMAN sets it.
- NEVER advance past G2 on your own. G3+ is human-gated.
- FORCE/CURRENT must be clamped far below 100 N during bring-up.
- Always respect joint ranges in hardware/LIMITS.md. Changes to src/safety
  need explicit human review.

## How to work
- Read your module CLAUDE.md + contracts/*.schema.json before coding.
- Write ONLY in your assigned module. Coordinate via contracts only.
- Done = named tests pass. Append a handoff note to STATE.md each turn.
- Record decisions as docs/adr/ files; never silently reverse one.

## Current state -> see STATE.md
```

---

## 7. Interface contracts

```jsonc
// contracts/hand_landmarks.schema.json  (MediaPipe Hands convention)
{ "side":"left|right", "landmarks":[[x,y,z], /* 21 pts: wrist,thumb1-4,index1-4,... */],
  "frame":"hand_base", "t":0.0 }

// contracts/l20_targets.schema.json
{ "side":"left|right",
  "joint_rad":[ /* 20 entries; idx 11-14 = 0.0 */ ],
  "active_idx":[0,1,2,3,4,5,6,7,8,9,10,15,16,17,18,19],
  "clamped":true, "t":0.0 }
```

---

## 8. First tickets (suggested order)

1. `eval/reference_solver/`: slow per-finger optimization oracle (ground truth for G0).
2. `solver-agent`: per-finger SEW alignment (4-finger 2+1 DoF, thumb 3+1) → G0 + timing.
3. `perception-agent`: MediaPipe Hands → normalized `hand_landmarks` → G0/G1 fixtures.
4. `sim-agent`: load `linkerhand-urdf` in PyBullet, FK tracking harness → G1.
5. `safety-agent`: inter-finger collision (XPBD) + force/current clamp → G2.
6. Integrate full loop in dynamic sim → G2.
7. **Human** reviews LIMITS.md, sets token → `comms-agent` real-L20 bring-up → G3.

---

## 9. Open items to confirm for your lab

- [ ] Human input source: MediaPipe Hands, a data glove, or Quest hand tracking?
- [ ] Confirm in `linkerhand-urdf` whether "tip" is one joint or two coupled joints
      in the model (affects how the distal alignment reads back in FK).
- [ ] Thumb opposition axis + zero calibration (thumb is the fiddly one).
- [ ] Sim choice: PyBullet (quick) vs MuJoCo (better contact) for G2.
- [ ] Map MediaPipe handedness/mirroring to the L20 left/right topics.
```
