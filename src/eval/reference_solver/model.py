"""Oracle FK adapter over the shared kinematics authority (gate G0).

Since the kinematics-agent refactor, the FK and the segment convention live in
ONE place — ``src/kinematics`` (ADR-0005). This module is now a thin adapter that
exposes the oracle's historical API (``L20Model``, ``FINGERS``, ``FingerSpec``,
the index constants) on top of ``src/kinematics.L20FK``, so the oracle's scipy
optimiser and the G0 tests keep working unchanged while sharing exactly the
solver/sim geometry.

CANONICAL CONVENTION (shared; see docs/adr/0003-segment-convention.md, amended by
Finding-1, and ADR-0006):

    r_prox = unit(P_b - P_a)          set by the BASE DoF
    r_dist = unit(fingertip - P_b)    to the PHYSICAL FINGERTIP (mimic curl IN)

Human landmark groups (positional, ADR-0003): for each finger's four landmarks
[a, b, c, d],  u_prox = unit(L_b - L_a),  u_dist = unit(L_d - L_b). See
``landmarks.py``.
"""
from __future__ import annotations

import numpy as np

from src.kinematics.conventions import (
    ACTIVE_IDX, RESERVED_IDX, N_JOINTS,
    FingerSpec, FINGERS, URDF_PATHS,
)
from src.kinematics.fk import L20FK

# Re-export so existing imports (`from .model import FINGERS`, etc.) keep working.
__all__ = [
    "ACTIVE_IDX", "RESERVED_IDX", "N_JOINTS",
    "FingerSpec", "FINGERS", "URDF_PATHS", "L20Model",
]


class L20Model:
    """URDF-backed FK exposing per-finger robot segment directions (oracle side).

    Delegates all kinematics to ``src/kinematics.L20FK`` (the single FK
    authority). Retains the oracle's method shapes:
    ``segment_dirs(spec, joint_values)``, ``finger_limits(spec)``,
    ``idx_to_joint(spec)``, ``set_cfg`` and the ``limits`` dict.
    """

    def __init__(self, side: str = "right"):
        if side not in URDF_PATHS:
            raise ValueError(f"side must be 'left'/'right', got {side!r}")
        self._fk = L20FK(side)
        self.side = side
        self.base_link = self._fk.base_link
        self.actuated_joint_names = self._fk.actuated_joint_names
        self.limits = self._fk.limits  # {joint_name: (lo, hi)} from the URDF

    # -- FK -----------------------------------------------------------------
    def set_cfg(self, joint_values: dict) -> None:
        self._fk.set_cfg(joint_values)

    def segment_dirs(self, spec: FingerSpec, joint_values: dict):
        """(r_prox, r_dist) unit vectors for the finger + config (fingertip
        r_dist, ADR-0003 Finding-1)."""
        rp, rd = self._fk.segment_dirs(spec, joint_values)
        return np.asarray(rp, float), np.asarray(rd, float)

    def finger_limits(self, spec: FingerSpec) -> dict:
        return self._fk.finger_limits(spec)

    def idx_to_joint(self, spec: FingerSpec) -> dict:
        return spec.idx_to_joint()


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < 1e-12:
        return np.array([0.0, 0.0, 1.0])
    return np.asarray(v, float) / n
