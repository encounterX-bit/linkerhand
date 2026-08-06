# eval/reference_solver — slow oracle (gate G0)

Ground-truth solver for the per-finger SEW retargeting. Numerically minimises the
canonical orientation-error objective `J` (see ADR-0003) over each finger's
actuated DoF, through the **real L20 URDF** forward kinematics (`yourdfpy`, so the
distal `mimic` coupling is honoured automatically). Deliberately slow; never used
in the hot path. The closed-form solver in `src/finger_retarget/` is validated
against this in `tests/g0_unit/`.

## Layout
- `model.py`      — `L20Model` URDF FK + `FINGERS` semantic map + segment dirs.
- `landmarks.py`  — 21 landmarks → per-finger `u_prox`, `u_dist`.
- `objective.py`  — canonical `J` (geodesic per-segment angle error).
- `oracle.py`     — `solve_oracle(landmarks, side)` → l20_targets dict.
- `synth_landmarks.py` — generates the G0 fixtures (flat/fist/pinch/point/thumbs_up,
  both hands) into `tests/g0_unit/fixtures/`. Run as a module to regenerate.

## Usage
```python
from eval.reference_solver import solve_oracle, L20Model
m = L20Model("right")                      # reuse across calls (URDF load is slow)
out = solve_oracle(landmarks_21x3, "right", model=m)
# out: {side, joint_rad[20] (idx 11-14=0), active_idx, clamped=True, t}
```

## Env
Needs `yourdfpy`, `scipy`, `numpy` (see `eval/requirements.txt`). The oracle is
grid-validated: its per-finger minimum matches a dense brute-force search.
