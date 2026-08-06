# Ticket: Restore Closed Form Across the Whole Solver (audit → convert)

**Supersedes** the thumb-only closed-form ticket — run this instead.
**Module (write only):** `src/finger_retarget/`.
**Gate:** correctness-gated. Sim/CPU only.
**Depends on (read-only):** `src/kinematics/` (FK, joint axes, mimic ratios, ADR-0003
fingertip target), `eval/reference_solver/` (the fingertip oracle — ground truth,
**do not modify**).
**Done =** every subproblem is closed-form (or, where provably necessary, fixed-cost
analytic), each matches the oracle to the current tolerance, should-not-change set unmoved.

---

## Background (what's actually broken)
Finding 1 (fingertip target) broke closed form for the **distal solve on all five
fingers**, not just the thumb:
- **Non-thumb distal (tip, 1-DoF):** currently a 1-D *search* (cheap, so it never tripped
  a timing alarm — but it's not closed-form).
- **Thumb distal (tip):** currently an iterative joint solve (fixed-point + grid).
- **Proximal (base) solves:** Paden-Kahan closed-form, believed intact — to be verified.

Out of scope (iterative *by design*, not regressions): the XPBD safety filter and the
scipy oracle. This ticket is the retargeting solver only.

## PHASE 1 — audit (classify before changing anything)
Enumerate every subproblem the solver runs and build a matrix:

| subproblem | DoF | current method | geometric reason | candidate reduction |
|---|---|---|---|---|
| non-thumb proximal | 2 | closed (PK)? | verify | — |
| non-thumb distal | 1 | 1-D search | coupled pip+dip | see below |
| thumb proximal | 3 | closed? | verify | — |
| thumb distal | 1 | iterative | coupled, non-collinear axes | polynomial |

Confirm the proximal solves are still genuinely closed-form (no silent fallback crept in).

## PHASE 2 — convert (exploit the structure per subproblem)
- **Non-thumb distal — likely the easy win.** pip and dip flexion axes are parallel and
  dip = 0.8917·pip, so for *direction* alignment the two rotations compose into a single
  rotation by the combined angle θ(1+0.8917) about the common flexion axis → the fingertip
  direction traces a **circle** → **Paden-Kahan subproblem-1**, clean closed form.
  **First verify the axes are actually parallel in the URDF.** If they are, this is direct.
  If not exactly, fall back to the polynomial reduction below.
- **Thumb distal — the hard one.** Axes are non-collinear, so the fingertip traces a
  non-circular curve. Apply `t = tan(θ/2)` to reduce the alignment condition to a
  polynomial; degree ≤ 4 → analytic radical roots (closed form); select by objective +
  tip limits. If degree > 4 / intractable → a **fixed-cost analytic** fallback
  (hard-capped Newton from an analytic seed; deterministic, bounded — not a search).
- **Proximal:** leave the closed-form base solves untouched.

## Hard constraint (uniform — this bounds "at all costs")
Per subproblem:
- **Match the fingertip oracle** to the **current method's tolerance** (worst over ≥3000
  random configs/side ≤ today's worst, not looser).
- **Do not change the target** (ADR-0003 fingertip), **do not drop/simplify the mimic
  coupling**, **do not loosen** any tolerance or test to make a formula pass. The geometry
  is messy because the target is correct — solve the messy one.
- If a correct closed form isn't reachable for a given subproblem, keep the correct
  fixed-cost/search **for that subproblem only** and report why. Never ship a wrong
  formula anywhere.

## Acceptance
- The audit matrix, before→after, committed: each subproblem closed-form, or fixed-cost
  analytic where provably necessary, with the reason and (for polynomial cases) the degree.
- Every converted subproblem matches the oracle to the current worst-case tolerance,
  both hands, ≥3000 configs each.
- **Should-not-change set:** oracle unchanged; G0 26/26; proximal numbers untouched;
  reachable round-trip ~1e-7; reserved idx 11–14 = 0; targets/mimic unchanged; hardware
  untouched. Distal residual numbers may shift only as the closed forms match the oracle
  (they should match the search's output, so expect ~no change).
- Timing reported (expect faster) — but correctness is the gate, not speed.
- ADRs per converted subproblem (method, derivation, root-selection / fallback reason).

## On finish
Update `STATE.md`: the audit matrix, the method per subproblem, oracle-match worst-case
numbers vs the prior method, timing, and explicit confirmation of every should-not-change
invariant. If any subproblem could not reach a correct closed form within the constraint,
say so and keep its correct method.

## Context to load
root, `src/finger_retarget/`, `src/kinematics/` (FK + axes + mimic + target),
`eval/reference_solver/` (read), ADR-0003, the ADR recording the iterative/search distal
solves, `STATE.md`, this ticket.
