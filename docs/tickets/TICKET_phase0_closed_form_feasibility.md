# TICKET: Phase-0 "Make-or-Break" — Does the Linker Hand admit an exact closed-form SEW-Mimic-style retargeting solution?

**Type:** Research spike / feasibility analysis
**Priority:** P0 (blocks paper framing and all downstream work)
**Environment:** Simulation only. **Do not touch hardware.**
**Owner:** Claude Code (autonomous), reporting to human researcher.

---

## 0. Context (read first)

We have ported the SEW-Mimic retargeting logic (closed-form geometric retargeting via Paden-Kahan / ik-geo subproblem decomposition, with a provable global-optimality guarantee) from a 7-DoF humanoid **arm** onto a **Linker Hand** dexterous hand. SEW-Mimic's optimality proof rests on two structural assumptions:

1. **Consecutive joint axes are mutually perpendicular** (so Subproblem 1/2 align "limb" proxy vectors exactly).
2. **Every joint used in the solution is independently actuated** (no passive/coupled DoF the solver cannot command).

Fingers very often violate **both**: flexion joints (MCP-flex, PIP, DIP) are typically **parallel**, not perpendicular, and the Linker Hand is a **hybrid active+passive architecture** with linkage/tendon coupling (e.g., DIP coupled to PIP; passive DoF). The competing methods (GeoRT, Dex-Retargeting, ByteDexter, Kilohertz-Safe) all sidestep this with learning, test-time optimization, or QP. Our only defensible contribution is a *genuinely closed-form, exact, optimization-free* solver. **This ticket decides whether that claim is true, false, or salvageable as a hybrid.**

---

## 1. The single question to answer

> Do the Linker Hand finger chains satisfy the structural conditions under which the ported SEW-Mimic subproblem decomposition yields an **exact, globally-optimal, non-iterative** solution — and if not, how large is the optimality gap introduced by (a) non-perpendicular / parallel consecutive axes and (b) passive/coupled DoF?

Answer it with **numbers and plots**, not prose. The output is a GO/NO-GO decision per Section 6.

---

## 2. Scope

**In scope:** repo discovery; per-finger kinematic-structure analysis; an audit confirming the solver is actually closed-form (not secretly iterative); an empirical residual study across the human-hand workspace in sim; a coupling-feasibility study.

**Out of scope:** hardware, teleoperation UX, policy learning, the safety filter, the arm. Those are later phases. Do not implement new retargeting methods — only analyze the existing one.

---

## 3. Tasks

### 3.1 Repo discovery (do this before assuming anything)
- Print the repo tree (depth 2–3). Identify and report the paths to: the hand model (URDF/MJCF/XML), the retargeting solver entry point, the per-joint axis/frame definitions, any coupling/mimic/transmission definitions, and the human-keypoint input format the solver expects.
- Detect which **Linker Hand variant** is modeled (O6 / O7 / L10 / L20 / L25 / L30 — differ in active vs passive DoF). Report active-DoF count, passive-DoF count, and number of fingers/joints from the model file itself, not from memory.
- If any of {hand model, solver, a pose dataset} cannot be found, **stop and report exactly what's missing** rather than fabricating it.

### 3.2 Confirm the solver is actually closed-form (claim audit)
- Statically inspect the solver: confirm there is **no** iterative optimizer, no Jacobian pseudo-inverse loop, no gradient descent, no test-time CMA/SLSQP/QP under the hood. List any iteration or `while`/`for`-until-converge constructs.
- Instrument it: log per-call wall-clock time, and any internal iteration counts. Closed-form should be sub-millisecond per finger with a fixed, bounded op count.
- Report: is the implemented logic genuinely SP1/SP2-style closed form, or has it quietly degraded into approximate/iterative IK? (This alone can flip the decision.)

### 3.3 Per-finger structural analysis
For each finger (including the thumb, analyzed separately — it's the worst case):
- Extract each joint's rotation axis, expressed in a common frame at a neutral pose.
- Compute the angle between **consecutive** axes. Classify each consecutive pair as perpendicular (~90°±tol), parallel (~0°/180°±tol), or oblique. Tolerance: 5°.
- Build the **actuation/coupling map**: for each joint, is it active, passive, or coupled (and to what, with what ratio)? Pull mimic-joint tags from the model and any transmission coupling from the driver/config.
- Compare the per-finger **independent actuated DoF** against the **dimensionality of the retargeting target** (fingertip position? keyvector set? orientation frame?). State whether each finger's target is over-, exactly-, or under-determined.
- Flag degeneracy regions: where parallel-flexion axes or joint limits cause solution-count changes or gimbal-lock-style singularities (cf. SEW-Mimic's perpendicular-wrist gimbal-lock remark).

### 3.4 Empirical residual study (the decisive test)
This mirrors SEW-Mimic's own optimality check: a provably-exact solver should return alignment error at floating-point precision.
- Assemble an input set of **human hand poses spanning the workspace**. Prefer a real source already in the repo (recorded teleop logs, DexYCB/ARCTIC/MANO/MediaPipe trajectories). If none exists, generate a dense, reproducible grid + random sampling of human finger configurations within published human ROM, and clearly label it as synthetic. Target ≥ 5,000 poses.
- Run the existing solver on every pose. For each finger, log the retargeting **residual** in both:
  - the orientation/cosine metric the solver optimizes (SEW-Mimic's `µc`/`µm`, expected ~1e-12 if exact), and
  - a physical proxy (fingertip / keyvector error in mm).
- Report residual distributions (median, IQR, 95th pct, max) per finger and overall, with histograms and a residual-vs-workspace heatmap. Use a Kruskal-Wallis test across fingers if comparing.
- **Interpretation:** residual at FP precision everywhere ⇒ closed-form holds. Residual systematically nonzero — especially correlated with parallel-axis or coupled-joint regions ⇒ the "exact closed-form" claim is false as written.

### 3.5 Coupling-feasibility study
- Take the raw joint angles the solver computes (as if all DoF were independent), then **project them onto the hand's feasible coupled manifold** (apply the real mimic/transmission constraints).
- Measure the resulting **error inflation** (post-projection residual minus pre-projection residual), per finger, across the workspace.
- Quantify how cheap a correction would need to be to recover near-zero error after projection (e.g., does a single 1-D line search or one Newton step suffice? report the achievable residual and op count).

---

## 4. Deliverables (write to `analysis/phase0/`)
1. `kinematic_structure_report.md` + `kinematic_structure.json` — Section 3.1 + 3.3 results (axis table, perpendicular/parallel classification, coupling map, DoF-vs-target verdict, degeneracy regions).
2. `solver_audit.md` — Section 3.2: closed-form-or-not verdict, timing, iteration counts.
3. `residual_study/` — Section 3.4: raw CSV, summary stats, histograms, workspace heatmaps.
4. `coupling_feasibility.csv` + a short plot — Section 3.5: error inflation and correction cost.
5. **`DECISION.md`** — one page: the GO/NO-GO verdict per Section 6, the three headline numbers, and a 3-bullet rationale a reviewer would accept.

---

## 5. Acceptance criteria
- All five deliverables exist and are populated from the **actual repo + sim**, not placeholders.
- Every quantitative claim traces to a logged artifact (CSV/plot), reproducible via a single documented command.
- `DECISION.md` states exactly one of {GO-Analytical, GO-Hybrid, PIVOT} with the supporting numbers.
- Any missing input (model, dataset) is reported explicitly, not worked around silently.

---

## 6. Decision rubric (thresholds — adjust with the human if borderline)

| Outcome | Conditions | Paper framing it unlocks |
|---|---|---|
| **GO-Analytical** | Solver confirmed non-iterative; residual at FP precision (`µ` < 1e-6) across the workspace; per-finger fully actuated OR coupling projection adds negligible error (< ~0.5 mm / `µ` < 1e-4). | Lead with the closed-form analytical solver + an optimality statement that *explicitly* covers the finger axis structure. Strongest, cleanest claim. |
| **GO-Hybrid** | Residual nonzero but bounded; a **cheap** bounded correction (one 1-D search / single Newton step) recovers near-zero error after coupling projection; correction cost keeps total latency ≪ optimization baselines. | Contribution = "closed-form backbone + provably-bounded correction for coupled/underactuated hands." Arguably *more* novel than a clean port. Still a strong RA-L paper. |
| **PIVOT** | Solver is secretly iterative/approximate; OR residual large and uncontrolled; OR coupling makes targets unreachable across much of the workspace; OR optimality empirically false. | Stop competing on closed-form retargeting. Reframe toward tactile-in-the-loop retargeting or a hardware/system + policy-learning paper. Reuse all Phase-0 evidence as honest motivation. |

---

## 7. Notes / gotchas
- The **thumb** is the most likely to break the closed-form assumption — analyze it separately and weight the decision toward its result.
- Distinguish "the proof's assumptions are violated" (expected) from "the solver still happens to be exact anyway" (only true if the violated assumption didn't matter for that chain). The empirical residual study (3.4) is the arbiter, not the structural analysis alone.
- Do not soften a PIVOT verdict. A correct negative result here saves months and is itself a finding that makes the eventual paper honest.
