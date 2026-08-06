"""src/kinematics — the single sim-independent FK source of truth for the L20.

One pure (yourdfpy/analytic) forward kinematics, used by the oracle
(``eval/reference_solver``), the sim metric (``src/sim``), and the solver's
offline codegen (``src/finger_retarget/gen_constants``). The segment convention
(ADR-0003, Finding-1 endpoint) and all semantic joint/segment/landmark maps live
in ``conventions.py`` so they can never diverge. See ADR-0005 (FK authority) and
the ``kinematics-agent-refactor`` ticket.

Public API:
    L20FK            -- pure FK + per-finger segment directions + limits
    conventions      -- FINGERS, FingerSpec, joint/segment/landmark maps, TIP_LOCAL
"""
from .conventions import (
    ACTIVE_IDX, RESERVED_IDX, N_JOINTS, FINGER_ORDER, BASE_LINK,
    URDF_PATHS, FingerSpec, FINGERS, JOINT_NAME, LANDMARK_GROUP,
    SEGMENT_LINKS, TIP_LOCAL,
)
from .fk import L20FK

__all__ = [
    "L20FK",
    "ACTIVE_IDX", "RESERVED_IDX", "N_JOINTS", "FINGER_ORDER", "BASE_LINK",
    "URDF_PATHS", "FingerSpec", "FINGERS", "JOINT_NAME", "LANDMARK_GROUP",
    "SEGMENT_LINKS", "TIP_LOCAL",
]
