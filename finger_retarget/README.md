# src/finger_retarget — closed-form per-finger SEW solver (gate G0)

Maps 21 MediaPipe hand landmarks → 16 actuated Linker Hand L20 joint radians
(reserved idx 11–14 = 0). Pure function, no I/O, no hardware import, deterministic,
closed-form. Consumes `contracts/hand_landmarks.schema.json`, emits
`contracts/l20_targets.schema.json`.

## API
```python
from src.finger_retarget import retarget
out = retarget(landmarks_21x3, side="right")   # or "left"
# out = {side, joint_rad[20] (idx 11-14 = 0.0), active_idx, clamped=True, t}
```

## Method (per finger; see ADR-0001/0002/0003/0004)
- **Base** (proximal direction): non-thumb 2 axes via Paden–Kahan subproblem-2;
  thumb 3 CMC axes solved exactly via a two-plane tip-axis construction.
- **Distal** (single tip DoF): subproblem-1 (the exact 1-DoF angle minimiser).
- All outputs clamped to the real URDF ranges (`hardware/LIMITS.md`).

Reachable targets are solved **exactly** (J ≈ 0, ~3e-6 over 2000 round-trip poses
/side). Genuinely under-actuated (nearest-reachable) thumb targets fall to a
bounded grid over the redundant base DoF — sized to keep the worst-case full-hand
solve within the 3 kHz real-time budget (it matches the slow oracle to ~p95 2e-3
with a bounded worst case; reachable stays exact).

## Layout
- `solver.py`    — `retarget()` + per-finger solves (hot path).
- `geometry.py`  — scalar (3-tuple) rotation geometry + Paden–Kahan subproblems.
  Scalar (not numpy) because numpy's per-call overhead dominates for size-3
  vectors; this is what meets the 3 kHz budget.
- `constants.py` — AUTO-GENERATED baked PoE constants (axes, zero-pose segment
  directions, limits) per finger/side. Do not hand-edit.
- `gen_constants.py` — OFFLINE codegen for `constants.py` from the vendored URDF
  (`python -m src.finger_retarget.gen_constants`). Not imported at runtime.

## Performance
Full-hand solve (target CPU): representative p50 ≈ 63 µs, p99 ≈ 73 µs (≪ 333 µs);
worst-case (every finger on the under-actuated thumb path) p99 ≈ 273 µs (< 333 µs).
See `tests/g0_unit/test_solver_g0.py::test_timing_3khz_*`.
