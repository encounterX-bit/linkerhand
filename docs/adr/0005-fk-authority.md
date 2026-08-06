# ADR-0005: One sim-independent FK authority (`src/kinematics`)

Status: accepted (kinematics-agent-refactor, 2026-06-09)
Related: [ADR-0003 segment convention], [ADR-0006 fingertip distal], ticket
kinematics-agent-refactor

## Context

Before this refactor the L20 forward kinematics + segment convention were
restated in three places that could silently drift: the oracle's `L20Model`
(yourdfpy FK), the sim harness's `L20Kinematics` (PyBullet FK) and the solver's
`gen_constants.py` codegen. PyBullet and yourdfpy FK agreed to ~1e-8, but nothing
*enforced* that, and the segment maps were copied by hand. Finding-1 (the
fingertip distal endpoint) had to change the segment definition in ONE place so
it could never diverge again.

## Decision

`src/kinematics` is the single, sim-independent FK authority:

- **FK authority = pure (yourdfpy / analytic), NOT PyBullet.** `L20FK` loads the
  vendored URDF via yourdfpy (mimics applied automatically); it exposes link
  transforms, per-finger `segment_dirs`, joint limits and mimic ratios.
- **PyBullet is no longer an FK authority.** It stays in `src/sim` for
  dynamics/contact (and the mimic-enforcement check) only. The sim's
  measurement/metric `segment_dirs` now delegates to `src/kinematics`.
- **One convention source.** `src/kinematics/conventions.py` holds the canonical
  DoF layout, per-finger `FingerSpec` (semantic idx -> driver joint), ADR-0003
  segment links, human landmark groups, and the mesh-derived fingertip offset
  (`TIP_LOCAL`). The oracle (`eval/reference_solver/model.py`), the sim
  (`src/sim/conventions.py`) and the solver codegen
  (`src/finger_retarget/gen_constants.py`) all import it.

## Consequences

- Extraction proven faithful: `L20FK` (legacy DIP-origin) `segment_dirs` matched
  the historical oracle FK to 0.0 over 600 random in-limit configs/side, and
  `L20FK` link transforms agree with PyBullet to <1e-7 pos / <1e-6 rot
  (`tests/g1_kinematic/test_fk_authority.py`).
- No FK logic is duplicated outside `src/kinematics`; conventions read from one
  source; the import swaps in sim and oracle are complete.
- Joint limits and mimic ratios are still read from the URDF at load (not
  hardcoded). The one mesh-derived constant (`TIP_LOCAL`) is baked offline so the
  FK and the solver hot path never load a mesh.
