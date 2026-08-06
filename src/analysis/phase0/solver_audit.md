# Phase-0 §3.2 — Solver Claim Audit: is it actually closed-form?

**Verdict: NO — not as a whole.** The solver is a **hybrid**: an exact closed-form
**base/proximal** solve (Paden–Kahan subproblem-2) wrapped around an **iterative,
data-dependent, unbounded-op-count distal solve** (1-D minimisation for non-thumb;
fixed-point + brute-force grid search for the thumb). The "exact, globally-optimal,
non-iterative" property holds **only for the per-finger base direction**, not for
the fingertip (distal) direction and not for the thumb overall.

This is not a regression or a bug — it is forced by the hand's structure (see
`kinematic_structure_report.md`): the coupled distal is a **parallel-axis** pair,
which has no Paden–Kahan subproblem-1 closed form. The code says so explicitly.

Source of truth: `src/finger_retarget/solver.py` (static read) +
`analysis/phase0/discover.py` (timing/path instrumentation).
Reproduce: `.venv/bin/python analysis/phase0/discover.py`.

---

## 1. Static inspection — iterative constructs found

| Path | Function (solver.py) | Construct | Closed-form? |
|---|---|---|---|
| Non-thumb **base** | `_two_axis_base` → `subproblem2` | Paden–Kahan SP2, branch-min | **YES (exact)** |
| Non-thumb base, *unreachable* | `_two_axis_base` | `for _ in range(4)` coordinate SP1 steps → nearest-reachable | **NO (bounded iterative)** |
| Non-thumb **distal (tip)** | `_tip_solve` | 5-sample grid bracket + `_brent_min` (`_TIP_ITERS=7` safeguarded-parabolic / golden) | **NO (iterative 1-D min)** |
| Thumb **base+tip (joint)** | `_solve_thumb` | `for _ in range(_THUMB_CD_ITERS=5)` fixed point on `cd`; each iter calls `_tip_solve` | **NO (fixed-point iteration)** |
| Thumb degenerate / near-parallel | `_thumb_grid` | 13-sample × 2-branch grid + `_THUMB_GRID_REFINE=18` golden-section steps | **NO (brute-force search)** |
| Thumb r_prox-unreachable | `_thumb_nearest` | 9-sample base grid + per-sample SP1 | **NO (grid search)** |

Confirmed **absent**: no Jacobian pseudo-inverse, no gradient descent over the full
hand, no CMA/SLSQP/QP, no scipy/optimiser import in `src/finger_retarget`. The
iteration is all *bounded-iteration* 1-D scalar search (Brent/golden) and *grid
search* — not a generic IK optimiser, but **not** closed form either.

The module docstring and the `_solve_thumb` / `_tip_solve` comments state this
directly: *"the Finding-1 fingertip is a 1-DoF CURVE … so there is no Paden-Kahan
subproblem-1 closed form"* and *"the old two-plane closed form (ADR-0004) … the
Finding-1 fingertip curve breaks that, hence the numerical joint solve."* See also
ADR-0006 and ADR-0007.

## 2. Is the op count fixed and bounded? — NO (data-dependent)

Instrumented over 500 full-hand `retarget()` calls (5 G0 poses × 2 sides × 50
variants; 0.002 m landmark jitter):

- `_tip_solve` (1-D Brent minimiser) invoked **19,195** times — i.e. ~38 per hand
  call. Non-thumb accounts for a fixed 4/call (2,000 total); the **thumb alone
  drives ~17,195** because the fixed point and the grid each re-run the 1-D tip
  solve many times.
- The thumb fell through to the **brute-force `_thumb_grid` on 465/500 = 93.0%** of
  solves. The advertised "closed-form fixed-point on `cd`" path is therefore **not
  the operative path** for most inputs in this set; the operative path is grid search.
  *(Caveat: this set is the synthetic G0 poses, whose thumb is partly outside the L20
  reachable cone — STATE.md/Step-1 note. The reachable-vs-unreachable split is exactly
  what the §3.4 residual sweep will quantify on a workspace-spanning set.)*
- `_thumb_nearest`: 0 hits here (r_prox stayed reachable).

The number of objective evaluations per call thus **varies with the input pose**
(reachable → few; near-parallel/under-actuated → grid). A genuine closed form has a
*fixed* op count regardless of input. This one does not.

## 3. Timing

Full-hand `retarget()` over the 500-call set (single thread, this machine):

| p50 | p95 | p99 | max | mean |
|---|---|---|---|---|
| 834.7 µs | 912.8 µs | 975.1 µs | 1057.7 µs | 797.7 µs |

- This is **~0.17 ms/finger averaged**, but the cost is **not** uniform: the four
  non-thumb fingers are cheap and near-constant; the **thumb dominates** (fixed-point
  + 13×2 grid + 18 golden ≈ tens of `_tip_solve` evals). The ticket's "sub-millisecond
  per finger with a fixed, bounded op count" is met on *latency per finger* but
  **fails the "fixed, bounded op count" criterion** — the thumb's cost is input-driven.
- Consistent with ADR-0007, which already retired the closed-form-era 3 kHz budget
  *because* the thumb distal became iterative. The numbers above are higher than
  ADR-0007's quoted p50≈175 µs because this batch's jittered synthetic thumbs push
  ~93% into the grid path (worst case), and timing is machine-specific.

## 4. Bottom line for the decision

The "genuinely closed-form, optimization-free" claim is **true for the base/proximal
alignment only** and **false for the distal/fingertip alignment and for the thumb**.
Per the ticket's own warning ("This alone can flip the decision"), a clean
**GO-Analytical is already off the table** unless the empirical residual study (§3.4)
shows the iterative distal returns FP-precision residuals everywhere anyway *and* we
are willing to reframe "closed-form" to mean "closed-form base + a bounded 1-D search
distal." The realistic candidates are **GO-Hybrid** (closed-form backbone + a
provably-bounded cheap correction — which is essentially what the code already is) or
**PIVOT** (if residuals are large/uncontrolled or coupling makes targets unreachable
across much of the workspace). The §3.4 residual sweep is the arbiter.
