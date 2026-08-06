# Ticket: `kinematics-agent` — Shared FK Source of Truth + Finding-1 Fix

**Modules:** creates `src/kinematics/`; **authorized to edit** `eval/reference_solver/`,
`src/sim/`, and `src/finger_retarget/` (distal subproblem only).
**Gate:** foundation refactor — run SERIALLY on `main`/a refactor branch BEFORE the
safety + G2 fan-out. Not parallel with feature work.
**Depends on:** the vendored URDF, ADR-0003, the existing yourdfpy oracle FK and
PyBullet sim FK (which already agree to ~1e-8).
**Done =** acceptance invariants below all hold.

> **This ticket is the explicit, orchestrator-authorized exception to "write only
> your module."** It is a coordinated cross-module refactor. Do the steps in order,
> commit after each, and do not expand scope beyond what's listed. If any
> should-be-unchanged invariant breaks, STOP and report — do not "fix" it by
> adjusting tests.

---

## Goal
One sim-independent FK, used by sim, oracle, safety, and the solver, with the
Finding-1 distal definition fixed in that one place so it can never diverge again.

## Decisions already made (do not re-litigate; implement them)
- **FK authority = pure (yourdfpy / analytic), NOT PyBullet.** PyBullet stays in
  `src/sim/` for **dynamics/contact only**; it is no longer an FK authority.
- **`r_dist` runs to the PHYSICAL FINGERTIP** (tip of the last distal link,
  including the mimic'd curl), to match the human `u_dist = TIP − PIP`. This is the
  Finding-1 fix. (Human confirms the ADR-0003 wording.)

## Steps
1. **Create `src/kinematics/`.** Promote the pure FK here: `fk(joint_rad) -> link
   transforms` and `segment_dirs(joint_rad) -> {r_prox, r_dist} per finger`,
   mimic-aware (the 0.8917 / 1.1619 ratios live here now). Move the shared
   conventions (joint map, limits, mimic ratios, ADR-0003 segment map — currently
   in `src/sim/conventions.py`) into / beside this module so there is ONE source.
   First verify the new FK matches the existing oracle FK to ~1e-8 on the **old**
   segment definition (proves the extraction is faithful) — then apply the
   fingertip fix.
2. **Finding-1 fix.** Change `segment_dirs` so `r_dist` is the direction to the
   fingertip link's tip, including the DIP/IP mimic curl — not the DIP origin.
   Update ADR-0003 to state the endpoint convention explicitly.
3. **Oracle.** Point `eval/reference_solver/` at `src/kinematics/`; regenerate the
   cached ground truth. Oracle distal numbers WILL change — expected.
4. **Solver distal re-derivation (`src/finger_retarget/`, distal only).** The
   fingertip direction is still a **1-DoF** function of the single tip command
   (DIP = ratio·tip), so re-derive the distal alignment to align the fingertip
   vector. **Leave the proximal subproblem untouched** (it's exact). Re-validate
   the solver against the NEW oracle (`matches-oracle` within ε). The thumb is the
   special case again (IP not PIP/DIP, ratio 1.1619) — verify it explicitly.
   - *Fallback (sanctioned):* if the coupled non-collinear axes don't yield clean
     closed form, a 1-D bounded scalar minimization over the single tip DoF (using
     `src/kinematics` FK) is acceptable — deterministic and well within the timing
     budget. Prefer closed-form; report which you used.
5. **Sim.** Swap `src/sim/` to use `src/kinematics/` for the **measurement / metric**
   FK (so measured `r_dist` matches the solver/oracle exactly). PyBullet now only
   provides dynamics. Re-run G1; re-measure the distal residual (expect movement).

## Acceptance — invariants that MUST hold
**Must NOT change (else the refactor leaked):**
- G0 still 26/26 green.
- Reachable round-trip still ~1e-7 rad (self-consistent under the new definition).
- New-FK ↔ oracle-FK transform agreement still ~1e-8.
- Proximal residual still ~exact.
- Hardware untouched: no `src/comms`, no `HW_ENABLE_TOKEN`, no actuation.

**Intended to change (and only these):**
- `r_dist` / distal segment definition (now fingertip-inclusive).
- Oracle distal ground truth (regenerated).
- Solver distal subproblem (re-derived) + its validation numbers.
- G1 distal residual numbers (re-measured; likely worse = more honest).

**Structural:** no FK logic duplicated outside `src/kinematics/`; conventions read
from one source; `import` swaps complete in sim and oracle.

## On finish
Update `STATE.md` with: what FK became canonical, the closed-form-vs-1D-search
choice for the distal solve, the before/after distal residual numbers, and ADRs
for the FK-authority decision and the ADR-0003 endpoint update. Confirm each
acceptance invariant explicitly in the handoff note.

## Context to load
root CLAUDE.md, ADR-0003, the URDF, `src/sim/{kinematics,conventions}.py`,
`eval/reference_solver/`, `src/finger_retarget/` (distal subproblem), this ticket.
