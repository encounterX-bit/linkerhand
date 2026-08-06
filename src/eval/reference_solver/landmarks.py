"""Human landmark -> per-finger segment unit vectors (gate G0).

Positional convention (ADR-0003): each finger owns four landmark indices
[a, b, c, d]; we read

    u_prox = unit(L_b - L_a)
    u_dist = unit(L_d - L_b)   (aggregate distal: the c landmark is collapsed)

For non-thumb fingers [a,b,c,d] = [MCP, PIP, DIP, TIP] so this is the ticket's
u_prox = PIP-MCP, u_dist = TIP-PIP. For the thumb [a,b,c,d] = [CMC, MCP, IP, TIP]
so u_prox is the metacarpal bone (CMC->MCP) and u_dist the MCP->TIP aggregate,
matching the robot's r_prox (set by the 3 CMC DoF) and r_dist (set by thumb_mcp).

Orientation only: outputs are unit vectors, so the mapping is invariant to the
overall scale of the landmark cloud (calibration-free).
"""
from __future__ import annotations

import numpy as np

from .model import FINGERS

FINGER_LANDMARKS = {name: spec.landmarks for name, spec in FINGERS.items()}


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < 1e-12:
        return np.array([0.0, 0.0, 1.0])
    return v / n


def finger_segment_dirs(landmarks, finger_name: str):
    """Return (u_prox, u_dist) unit vectors for one finger.

    Parameters
    ----------
    landmarks : array-like, shape (21, 3)  -- MediaPipe hand_base-frame points.
    finger_name : one of 'thumb','index','middle','ring','little'.
    """
    lm = np.asarray(landmarks, dtype=float)
    if lm.shape != (21, 3):
        raise ValueError(f"landmarks must be (21,3), got {lm.shape}")
    a, b, c, d = FINGERS[finger_name].landmarks
    u_prox = _unit(lm[b] - lm[a])
    u_dist = _unit(lm[d] - lm[b])
    return u_prox, u_dist


def all_segment_dirs(landmarks) -> dict:
    """{finger_name: (u_prox, u_dist)} for all five fingers."""
    return {name: finger_segment_dirs(landmarks, name) for name in FINGERS}
