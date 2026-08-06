"""Raw-estimator landmarks -> the ADR-0003 ``hand_base`` frame.

This is the perception-side analogue of the thumb-label trap: if perception puts
landmarks in a frame even slightly rotated from the one the solver/oracle assume,
*every* downstream orientation error is silently corrupted. So the construction
here is pinned to reproduce the oracle/solver frame.

ADR-0003 axis convention (same as the L20 URDF):

    +z = finger pointing direction (wrist -> fingertips)
    +y = radial side (toward the thumb)   [for the RIGHT hand]
    +x = palm normal, dorsal              (flexion curls fingers toward -x)
    origin = wrist (landmark 0)

The left hand is the mirror image of the right (the synthetic G0 fixtures mirror
the radial axis), so for ``side == "left"`` the radial reference is flipped; the
result is a right-handed coordinate frame whose point cloud has left-hand
chirality -- exactly matching the fixtures.

Why the **palm plane** (wrist + the four finger MCPs) and not the wrist->middle
vector: those five points are pose-invariant and, in the oracle frame, exactly
coplanar (x == 0). Building the basis from their best-fit plane therefore
reproduces the oracle frame *exactly* on every G0 fixture (see the
frame-convention test), while a single-landmark forward vector would tilt the z
axis by a few degrees. The transform is a pure rigid map (rotation about the
wrist), so it also exactly inverts any camera rotation/translation of a hand.
"""
from __future__ import annotations

import numpy as np

from .indices import INDEX_MCP, PALM_LANDMARKS, PINKY_MCP, WRIST

_EPS = 1e-9


def _unit(v: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > _EPS else fallback.copy()


def build_basis(landmarks, side: str):
    """Return ``(origin, R)`` mapping source-frame points into hand_base.

    ``R`` has the hand_base basis vectors (x, y, z) as its columns expressed in
    the source frame, so ``p_hand = R.T @ (p_src - origin)``.

    Raises ``ValueError`` if the palm is too degenerate to define a frame
    (callers treat this as a bad frame and hold the last good one).
    """
    lm = np.asarray(landmarks, dtype=float)
    if lm.shape != (21, 3):
        raise ValueError(f"landmarks must be (21,3), got {lm.shape}")
    if side not in ("left", "right"):
        raise ValueError(f"side must be 'left' or 'right', got {side!r}")
    if not np.all(np.isfinite(lm)):
        raise ValueError("landmarks contain non-finite values")

    origin = lm[WRIST].copy()
    palm = lm[list(PALM_LANDMARKS)]
    centroid = palm.mean(axis=0)

    # Palm-plane normal: the least-significant singular direction of the centred
    # palm points. Sign is resolved below via the right-handed construction.
    _, sv, vt = np.linalg.svd(palm - centroid)
    if sv[1] < 1e-6:  # points (near) collinear -> no plane
        raise ValueError("degenerate palm: MCPs collinear, cannot define frame")
    normal = vt[2]

    # Radial reference: index MCP -> pinky points to +y for the RIGHT hand;
    # mirror it for the left so the frame matches the mirrored fixtures.
    side_ref = lm[INDEX_MCP] - lm[PINKY_MCP]
    if side == "left":
        side_ref = -side_ref
    side_ref = side_ref - np.dot(side_ref, normal) * normal
    y_hat = _unit(side_ref, np.array([0.0, 1.0, 0.0]))

    # Forward (+z): orthogonal to normal and radial, pointing toward the fingers.
    forward_ref = centroid - origin
    z_hat = np.cross(normal, y_hat)
    z_hat = _unit(z_hat, np.array([0.0, 0.0, 1.0]))
    if np.dot(z_hat, forward_ref) < 0.0:
        z_hat = -z_hat

    # Dorsal palm normal (+x), forced right-handed: x = y x z.
    x_hat = _unit(np.cross(y_hat, z_hat), np.array([1.0, 0.0, 0.0]))
    # Re-orthonormalise y against the final x,z to kill round-off.
    y_hat = np.cross(z_hat, x_hat)

    R = np.column_stack((x_hat, y_hat, z_hat))
    return origin, R


def to_hand_base(landmarks, side: str) -> np.ndarray:
    """Express the 21 landmarks in the ADR-0003 hand_base frame.

    Pure rigid transform (rotate about the wrist); preserves metric scale. The
    solver is scale-invariant, so no normalisation is applied here.
    """
    lm = np.asarray(landmarks, dtype=float)
    origin, R = build_basis(lm, side)
    return (lm - origin) @ R
