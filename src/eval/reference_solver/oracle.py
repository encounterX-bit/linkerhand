"""Slow ground-truth solver (gate G0).

For each finger, numerically minimise the canonical objective J over that finger's
actuated DoF, bounded by the real URDF joint limits, via ``scipy.optimize`` with
multi-start to escape local minima. This is the oracle the closed-form solver is
checked against -- it is deliberately slow and never used in the hot path.

Output is a full l20_targets-shaped dict (20 joints, reserved idx 11-14 == 0,
clamped=True), consuming the hand_landmarks contract.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from .model import L20Model, FINGERS, ACTIVE_IDX, N_JOINTS
from .landmarks import finger_segment_dirs
from .objective import objective_J, DEFAULT_WEIGHTS

# multi-start grid (fractions of each joint's range) -- coarse but enough to find
# the global basin for these smooth 2-4 DoF problems.
_STARTS = (0.0, 0.25, 0.5, 0.75, 1.0)


def _finger_cost(x, joint_order, spec, model, u_prox, u_dist, weights):
    jv = dict(zip(joint_order, x))
    r_prox, r_dist = model.segment_dirs(spec, jv)
    return objective_J(r_prox, u_prox, r_dist, u_dist, weights)


def solve_finger_oracle(model, spec, landmarks, weights=DEFAULT_WEIGHTS,
                        n_restarts=3):
    """Return {semantic_idx: angle_rad} minimising J for one finger."""
    u_prox, u_dist = finger_segment_dirs(landmarks, spec.name)
    joint_order = spec.all_dof_joints              # base DoF..., tip
    idx_order = _dof_indices(spec)
    bounds = [model.limits[j] for j in joint_order]

    best_x, best_f = None, np.inf
    # deterministic multi-start: corners + center per dim is too many, so use a
    # small fixed set of scalar fractions broadcast across dims, plus all-mid.
    seeds = []
    for frac in _STARTS:
        seeds.append([lo + frac * (hi - lo) for (lo, hi) in bounds])
    # a couple of mixed seeds (tip extended vs flexed) help the coupled distal.
    mid = [0.5 * (lo + hi) for (lo, hi) in bounds]
    seeds.append(mid)

    rng = np.random.default_rng(0)  # deterministic
    for _ in range(n_restarts):
        seeds.append([rng.uniform(lo, hi) for (lo, hi) in bounds])

    for x0 in seeds:
        res = minimize(
            _finger_cost, np.asarray(x0, float),
            args=(joint_order, spec, model, u_prox, u_dist, weights),
            method="L-BFGS-B", bounds=bounds,
            options={"maxiter": 300, "ftol": 1e-12, "gtol": 1e-10},
        )
        if res.fun < best_f:
            best_f, best_x = res.fun, res.x

    return {idx: float(v) for idx, v in zip(idx_order, best_x)}


def _dof_indices(spec):
    """Semantic indices in the same order as spec.all_dof_joints."""
    j2i = {v: k for k, v in spec.idx_to_joint().items()}
    return [j2i[j] for j in spec.all_dof_joints]


def solve_oracle(landmarks, side="right", weights=DEFAULT_WEIGHTS, n_restarts=3,
                 model=None):
    """Full-hand oracle. Returns an l20_targets-shaped dict."""
    model = model or L20Model(side)
    joint_rad = [0.0] * N_JOINTS
    for spec in FINGERS.values():
        sol = solve_finger_oracle(model, spec, landmarks, weights, n_restarts)
        for idx, val in sol.items():
            joint_rad[idx] = val
    return {
        "side": side,
        "joint_rad": joint_rad,
        "active_idx": list(ACTIVE_IDX),
        "clamped": True,
        "t": 0.0,
    }
