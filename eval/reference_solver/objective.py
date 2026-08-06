"""Canonical per-finger orientation-error objective (gate G0).

    J(finger) = w_prox * angle(r_prox, u_prox) + w_dist * angle(r_dist, u_dist)

Geodesic (angular) error on the unit sphere -- NOT Euclidean -- so it is
calibration/scale-free. Both the oracle and the closed-form solver are scored by
this exact function in tests/g0_unit/ (test 1: J_cf <= J_oracle + eps).
"""
from __future__ import annotations

import numpy as np

DEFAULT_WEIGHTS = (1.0, 1.0)  # (w_prox, w_dist)


def angle_between(a, b) -> float:
    """Unsigned geodesic angle (radians) between two vectors."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    c = float(np.dot(a, b) / (na * nb))
    c = max(-1.0, min(1.0, c))
    return float(np.arccos(c))


def objective_J(r_prox, u_prox, r_dist, u_dist, weights=DEFAULT_WEIGHTS) -> float:
    w_prox, w_dist = weights
    return (
        w_prox * angle_between(r_prox, u_prox)
        + w_dist * angle_between(r_dist, u_dist)
    )
