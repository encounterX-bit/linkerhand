"""src/finger_retarget — closed-form per-finger SEW retargeting solver (G0).

Public API:
    retarget(landmarks, side) -> l20_targets dict   (pure, closed-form, clamped)

See docs/tickets/ticket-solver-agent-G0.md and ADR-0001/0002/0003.
"""
from .solver import retarget
from .constants import ACTIVE_IDX, RESERVED_IDX, N_JOINTS

__all__ = ["retarget", "ACTIVE_IDX", "RESERVED_IDX", "N_JOINTS"]
