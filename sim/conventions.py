"""L20 conventions for the G1 sim harness — re-exported from the ONE source.

Since the kinematics refactor (ticket kinematics-agent-refactor) the canonical
joint/segment/landmark maps live in ``src/kinematics.conventions``. This module
is now a thin re-export so the sim harness shares exactly the solver/oracle
convention (no second copy that could drift). See ADR-0005.
"""
from __future__ import annotations

from src.kinematics.conventions import (
    ACTIVE_IDX, RESERVED_IDX, N_JOINTS, FINGER_ORDER,
    BASE_LINK, JOINT_NAME, LANDMARK_GROUP, SEGMENT_LINKS,
)

__all__ = [
    "ACTIVE_IDX", "RESERVED_IDX", "N_JOINTS", "FINGER_ORDER",
    "BASE_LINK", "JOINT_NAME", "LANDMARK_GROUP", "SEGMENT_LINKS",
]
