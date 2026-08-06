# Ticket: `sim-agent` — G1 Kinematic Tracking Harness

**Module (write only):** `src/sim/` (+ `tests/g1_kinematic/`)
**Gate:** G1 (kinematic sim, NO dynamics)
**Depends on:** `src/sim/urdf/` (vendored L20), `contracts/{hand_landmarks,l20_targets}.schema.json`,
`src/finger_retarget` public `solve()` (import read-only — do NOT modify),
`hardware/LIMITS.md`, ADR-0003 (frame/segment convention),
`tests/g0_unit/fixtures/` (synthetic), and real recorded sequences from
`perception-agent` (arrive later, see §Sequencing).
**Done =** all of `tests/g1_kinematic/` green.

---

## Goal
Load the L20 in PyBullet (kinematic), drive it with solver outputs from landmark
sequences, and measure how well the *achieved* finger-segment directions track the
human's. This is the first measurement of retargeting **quality** — distinct from
the solver **correctness** that G0 already proved. Expect nonzero residuals on real
poses (under-actuation + the curved coupled distal section); the job is to measure
them, not to make them zero.

## Harness
1. Load `src/sim/urdf/` L20 in PyBullet. **Kinematic only** — use `resetJointState`,
   never `setJointMotorControl2` with forces. No gravity, no contact.
2. **ENFORCE MIMIC JOINTS MANUALLY.** PyBullet ignores URDF `<mimic>`. After setting
   the 16 active joints, set each mimic joint = ratio × parent (non-thumb
   `dip = 0.8917·pip`; thumb `ip/dip = 1.1619·thumb_mcp`, per `hardware/LIMITS.md`).
   Wrong mimic handling → wrong FK → silently wrong everything. Add an explicit test.
3. Per frame: `hand_landmarks` → `finger_retarget.solve()` → `l20_targets` →
   set 16 joints (+mimics) → FK → robot proximal/distal segment unit vectors per finger.
4. Compare to the human `u_prox, u_dist` from the same landmarks, in the ADR-0003
   frame. Metric = geodesic angle per segment.

## Tests (`tests/g1_kinematic/`)
1. **Reachable set.** Round-tripped configs (FK→landmarks→solve→FK) track within
   ≤ 0.01 rad. Hard pass.
2. **Real / unreachable set.** Replay recorded real sequences; report per-segment
   p50/p95 orientation error; assert ≤ `G1_RESIDUAL_THRESHOLD` (config). **If the
   threshold is unset, FAIL CLOSED** — a human sets the pass line; do not invent one.
3. **Limits & reserved.** Every commanded config within `hardware/LIMITS.md`; idx
   11–14 == 0; mimics within range; no NaN.
4. **Thumb axis confirmation.** Drive a synthetic pure-thumb-flexion sequence;
   assert the thumb **flexes toward the palm**, not primarily abducts — empirically
   confirms the `cmc_pitch`/`cmc_roll` label fix. Render frames for human review.
5. **Visualization.** Write an image/gif sequence (and per-frame CSV of target vs
   achieved directions, thumb included) to `tests/g1_kinematic/out/` for human eyeball.

## Notes
- Import `finger_retarget`'s public `solve()` as the system under test; do NOT edit
  it. If its interface is wrong, file back via `STATE.md` — don't patch cross-module.
- Reachable + synthetic + thumb-confirm tests need no perception output → do these
  first. The real-set residual test needs perception's recorded sequences in
  `tests/g1_kinematic/fixtures/real/` → run that subtask after they land.
- On finish: update `STATE.md`. If real-set residuals exceed threshold, do NOT
  advance — report the distribution so the human decides (solver tuning vs threshold).

## Context to load (nothing more)
root + `src/sim/CLAUDE.md`, the two contracts, this ticket, the URDF,
`hardware/LIMITS.md`, ADR-0003, fixtures.
