"""Retargeting-quality pipeline: landmarks -> solve -> FK -> per-segment error.

Glue between the human landmark stream, the (read-only) closed-form solver, and
the kinematic harness. The solver is imported as the SYSTEM UNDER TEST; this module
never modifies it. Its public entry point is ``retarget`` (the ticket calls it
``solve()`` — see STATE.md handoff note).
"""
from __future__ import annotations

import numpy as np

from src.finger_retarget import retarget  # read-only system under test

from .conventions import FINGER_ORDER, LANDMARK_GROUP


def _unit(v):
    v = np.asarray(v, dtype=float)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else np.array([0.0, 0.0, 1.0])


def geodesic_angle(a, b):
    """Geodesic (great-circle) angle in radians between two 3-vectors."""
    a, b = _unit(a), _unit(b)
    c = float(np.clip(np.dot(a, b), -1.0, 1.0))
    return float(np.arccos(c))


def human_segments(landmarks):
    """{finger: (u_prox, u_dist)} from MediaPipe landmarks (ADR-0003)."""
    lm = np.asarray(landmarks, dtype=float)
    out = {}
    for finger in FINGER_ORDER:
        a, b, _c, d = LANDMARK_GROUP[finger]
        out[finger] = (_unit(lm[b] - lm[a]), _unit(lm[d] - lm[b]))
    return out


def track_frame(kin, landmarks, side):
    """Run one frame end to end.

    landmarks -> retarget -> set_config(+mimics) -> FK segment dirs, then score the
    geodesic orientation error of each finger's proximal and distal bone against the
    human target from the *same* landmarks.

    Returns a dict:
        targets : the l20_targets dict from the solver
        applied : {joint_name: angle} actually set (drivers + mimics)
        human   : {finger: (u_prox, u_dist)}
        robot   : {finger: (r_prox, r_dist)}
        err     : {finger: (e_prox, e_dist)}  geodesic radians
    """
    targets = retarget(landmarks, side=side)
    applied = kin.set_config(targets["joint_rad"])
    robot = kin.segment_dirs()
    human = human_segments(landmarks)
    err = {
        f: (geodesic_angle(robot[f][0], human[f][0]),
            geodesic_angle(robot[f][1], human[f][1]))
        for f in FINGER_ORDER
    }
    return {"targets": targets, "applied": applied,
            "human": human, "robot": robot, "err": err}


def error_rows(record, frame=0, t=0.0):
    """Flatten a track_frame record into per-(finger,segment) rows for CSV/stats."""
    rows = []
    for f in FINGER_ORDER:
        for si, seg in enumerate(("prox", "dist")):
            u, r = record["human"][f][si], record["robot"][f][si]
            rows.append({
                "frame": frame, "t": t, "finger": f, "segment": seg,
                "err_rad": record["err"][f][si],
                "ux": u[0], "uy": u[1], "uz": u[2],
                "rx": r[0], "ry": r[1], "rz": r[2],
            })
    return rows
