# hardware/LIMITS.md

Source: `linker-bot/linkerhand-urdf` @ `src/sim/urdf/` (cloned Step 0, G0).
Model: `l20/right/linkerhand_l20_right.urdf` (left mirrors; see §Handedness).
All values are **radians**, read directly from the URDF `<limit lower/upper>` of
the *independent* (non-`mimic`) revolute joints.

> The URDF has **21 revolute joints**, but **5 are `mimic` joints** (fixed-ratio
> dependents). The **16 independent joints == the 16 actuated L20 DoF**. The
> mimic joints are how the distal "tip" coupling is encoded — see §Distal coupling.

## Actuated joint ranges (16 DoF + 4 reserved)

| idx | semantic name     | URDF joint (independent) | world axis @0 | min_rad | max_rad |
|-----|-------------------|--------------------------|---------------|---------|---------|
| 0   | thumb base        | thumb_cmc_pitch ⚠        | ~[1,0,0]      | 0.0     | 0.79    |
| 1   | index base        | index_mcp_pitch          | [0,1,0]       | 0.0     | 1.40    |
| 2   | middle base       | middle_mcp_pitch         | [0,1,0]       | 0.0     | 1.40    |
| 3   | ring base         | ring_mcp_pitch           | [0,1,0]       | 0.0     | 1.40    |
| 4   | little base       | pinky_mcp_pitch          | [0,1,0]       | 0.0     | 1.40    |
| 5   | thumb abduction   | thumb_cmc_roll ⚠         | ~[0,-.38,-.93]| 0.0     | 1.22    |
| 6   | index abduction   | index_mcp_roll           | [1,0,0]       | -0.17   | 0.17    |
| 7   | middle abduction  | middle_mcp_roll          | [1,0,0]       | -0.17   | 0.17    |
| 8   | ring abduction    | ring_mcp_roll            | [1,0,0]       | -0.17   | 0.17    |
| 9   | little abduction  | pinky_mcp_roll           | [1,0,0]       | -0.17   | 0.17    |
| 10  | thumb opposition  | thumb_cmc_yaw            | [1,0,0]       | 0.0     | 1.40    |
| 11  | RESERVED          | —                        | —             | 0.0     | 0.0     |
| 12  | RESERVED          | —                        | —             | 0.0     | 0.0     |
| 13  | RESERVED          | —                        | —             | 0.0     | 0.0     |
| 14  | RESERVED          | —                        | —             | 0.0     | 0.0     |
| 15  | thumb tip         | thumb_mcp                | [1,0,0]       | 0.0     | 1.05    |
| 16  | index tip         | index_pip                | [0,1,0]       | 0.0     | 1.57    |
| 17  | middle tip        | middle_pip               | [0,1,0]       | 0.0     | 1.57    |
| 18  | ring tip          | ring_pip                 | [0,1,0]       | 0.0     | 1.57    |
| 19  | little tip        | pinky_pip                | [0,1,0]       | 0.0     | 1.57    |

Reserved idx 11–14 are **always commanded 0.0** (per ROOT CLAUDE.md). They have
no URDF joint.

## Distal coupling — "tip" is ONE command, but TWO URDF joints (mimic)

The distal flexion is **not** a single URDF joint anywhere. It is always a driver
joint plus a fixed-ratio `mimic` dependent. This is what makes "PIP and DIP
collapse to one tip command" physically true on the L20 (ADR-0002):

- **Non-thumb fingers (index/middle/ring/little):**
  driver = `*_pip` (idx 16–19), dependent `*_dip = 0.8917 · *_pip` (mimic, offset 0).
  At pip = 1.57 → dip = 1.40, exactly the dip URDF limit — the ratio is tuned so
  both saturate together.
- **Thumb:**
  driver = `thumb_mcp` (idx 15), dependent `thumb_ip` (left) / `thumb_dip` (right)
  `= 1.1619 · thumb_mcp` (mimic, offset 0). The thumb's MCP and IP curl as one
  unit; that coupled curl IS the thumb "tip" DoF.

**FK / oracle must apply these mimic ratios** when computing distal link
directions. Underlying dependent-joint limits (for reference / safety cross-check):
non-thumb `*_dip` ∈ [0, 1.40]; thumb `ip/dip` ∈ [0, 1.22]. The driver ranges in
the table above already keep dependents in-range (1.57·0.8917=1.40; 1.05·1.1619=1.22).

## ⚠ Joints that could NOT be confidently mapped (need SDK confirmation)

The four non-thumb fingers map 1:1 and unambiguously (axis + parent/child + mimic
all agree). The **thumb** is the only ambiguity:

- `thumb_cmc_yaw` → **opposition (idx 10)** — confident. It is the most proximal
  CMC joint and rotates about the palm-normal (world ~[1,0,0]), i.e. swings the
  thumb across the palm = opposition. Largest range (1.40).
- `thumb_mcp` → **tip (idx 15)** — confident *given the mimic*: it is the only
  thumb joint that drives a coupled distal (`ip/dip` mimics it), so it must be the
  "coupled distal flexion" DoF. (Note: the root CLAUDE.md labels idx 0 as the
  thumb "MCP flexion"; the URDF's single MCP joint is coupled to the IP, so MCP
  flexion physically presents as the *tip* DoF here, not idx 0.)
- `thumb_cmc_pitch` vs `thumb_cmc_roll` → **base (idx 0) vs abduction (idx 5)**:
  **AMBIGUOUS.** Both are CMC DoF. I assigned base←`thumb_cmc_pitch` (rotates
  about world ~[1,0,0], flexion-like) and abduction←`thumb_cmc_roll` (off-axis
  twist ~[0,-.38,-.93], spread-like), but the URDF alone cannot prove which the
  L20 SDK calls "base" vs "abduction." **Ranges for both are recorded above, so
  clamping is safe regardless of which way the assignment resolves.** Confirm
  against the SDK joint-command order (`linkerhand-ros-sdk`) before G1.

No xacro files exist in the repo — the L20 URDFs are already plain URDF, so no
expansion was needed.

## Handedness (left vs right)

Ranges are identical magnitude on both hands. Differences observed:
- Base link: `base_link` (right) vs `hand_base_link` (left).
- Thumb distal joint name: `thumb_dip` (right) vs `thumb_ip` (left). Same mimic.
- Sign/axis flips on `thumb_cmc_yaw` (axis `[1,0,0]` right vs `[-1,0,0]` left) and
  the thumb CMC chain origin ordering — i.e. abduction/opposition senses mirror.
  The solver must apply side-correct signs (G0 test 7, handedness).

## Hardware bring-up clamps (G3) — HUMAN-SET, do not fill from an agent

- FORCE / CURRENT cap: TODO  (start FAR below 100 N grip max)
- speed cap: TODO
- e-stop: TODO (physical + software)
