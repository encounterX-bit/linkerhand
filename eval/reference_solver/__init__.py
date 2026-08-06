"""Reference (oracle) solver for L20 per-finger SEW retargeting — gate G0.

This package is the SLOW ground-truth optimizer used to validate the closed-form
solver in ``src/finger_retarget/``. It is intentionally not in the hot path.

Public API:
    L20Model            -- URDF-backed forward kinematics + segment directions
    finger_segment_dirs -- human u_prox/u_dist from 21 landmarks
    objective_J         -- canonical per-finger orientation-error objective
    solve_oracle        -- scipy.optimize ground-truth solver

The CANONICAL conventions (shared, by spec, with the closed-form solver and the
G0 tests — see docs/adr/0003-segment-convention.md) live in ``model.py``.
"""

from .model import (
    L20Model,
    FINGERS,
    ACTIVE_IDX,
    RESERVED_IDX,
    FingerSpec,
)
from .landmarks import finger_segment_dirs, FINGER_LANDMARKS
from .objective import objective_J, angle_between
from .oracle import solve_oracle

__all__ = [
    "L20Model",
    "FINGERS",
    "ACTIVE_IDX",
    "RESERVED_IDX",
    "FingerSpec",
    "finger_segment_dirs",
    "FINGER_LANDMARKS",
    "objective_J",
    "angle_between",
    "solve_oracle",
]
