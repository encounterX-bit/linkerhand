"""Synthetic landmark generation for kinematic tests (no perception needed).

Two generators:
  - :func:`landmarks_from_dirs` builds a valid 21-point MediaPipe landmark array
    whose per-finger ``(u_prox, u_dist)`` equal prescribed directions. Only
    directions matter to the convention (scale/translation invariant, ADR-0003), so
    absolute anchor placement is arbitrary.
  - :func:`random_config` / :func:`config_to_landmarks` drive the FK->landmarks half
    of the reachable-set round-trip: a real robot config -> its segment directions
    -> landmarks that exactly request those directions.
"""
from __future__ import annotations

import numpy as np

from .conventions import FINGER_ORDER, LANDMARK_GROUP, ACTIVE_IDX, N_JOINTS

_L1 = 0.040  # synthetic proximal bone length (m); arbitrary, directions only
_L2 = 0.038  # synthetic distal bone length (m)


def landmarks_from_dirs(dirs, anchors=None):
    """21x3 landmarks with per-finger u_prox/u_dist == ``dirs[finger]``.

    dirs    : {finger: (u_prox(3), u_dist(3))}
    anchors : optional {finger: P_a(3)}; defaults spread the fingers laterally so
              the cloud is non-degenerate (irrelevant to per-finger directions).
    """
    lm = np.zeros((21, 3), dtype=float)
    for fi, finger in enumerate(FINGER_ORDER):
        a, b, c, d = LANDMARK_GROUP[finger]
        u_prox = _unit(dirs[finger][0])
        u_dist = _unit(dirs[finger][1])
        Pa = np.asarray(anchors[finger], float) if anchors else np.array([0.02 * fi, 0.0, 0.0])
        Pb = Pa + _L1 * u_prox
        Pd = Pb + _L2 * u_dist
        Pc = Pb + 0.5 * _L2 * u_dist  # collapsed landmark; on the distal bone
        lm[a], lm[b], lm[c], lm[d] = Pa, Pb, Pc, Pd
    return lm


def random_config(rng, limits, n=None):
    """A random in-limits 20-vector joint_rad (reserved idx left 0.0)."""
    q = [0.0] * N_JOINTS
    for idx in ACTIVE_IDX:
        lo, hi = limits[idx]
        q[idx] = float(rng.uniform(lo, hi))
    return q


def config_to_landmarks(kin, joint_rad):
    """Set ``joint_rad`` on the harness and synthesize landmarks requesting the
    resulting robot segment directions (the FK->landmarks leg of a round-trip)."""
    kin.set_config(joint_rad)
    dirs = kin.segment_dirs()
    return landmarks_from_dirs(dirs)


def _unit(v):
    v = np.asarray(v, dtype=float)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else np.array([0.0, 0.0, 1.0])
