"""Synthetic human-hand landmark generator for G0 fixtures.

A small, self-contained human-hand forward model (NOT derived from the robot, so
the oracle/solver are not graded against their own FK). Produces MediaPipe-style
21-landmark sets in a ``hand_base`` frame using the same axis convention as the
L20 URDF:  +z = finger pointing direction, +y = toward the radial (thumb) side,
+x = palm normal (flexion curls fingers toward -x).

Running this module rewrites tests/g0_unit/fixtures/*.json.
"""
from __future__ import annotations

import json
import os

import numpy as np

# MediaPipe Hands landmark indices
WRIST = 0
# finger -> (mcp, pip, dip, tip) landmark indices (thumb: cmc, mcp, ip, tip)
GROUPS = {
    "thumb": (1, 2, 3, 4),
    "index": (5, 6, 7, 8),
    "middle": (9, 10, 11, 12),
    "ring": (13, 14, 15, 16),
    "little": (17, 18, 19, 20),
}

# canonical human geometry (metres). y>0 = radial (index/thumb) side.
PALM_LEN = 0.09  # wrist -> finger MCP row
MCP_Y = {"index": 0.022, "middle": 0.006, "ring": -0.012, "little": -0.030}
# (proximal, middle, distal) bone lengths
BONES = {
    "index": (0.040, 0.025, 0.020),
    "middle": (0.045, 0.028, 0.022),
    "ring": (0.040, 0.026, 0.021),
    "little": (0.032, 0.020, 0.018),
}
# thumb (metacarpal, proximal, distal)
THUMB_CMC = np.array([0.0, 0.035, 0.030])     # radial + slightly forward
THUMB_BONES = (0.045, 0.032, 0.025)


def _rx(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _ry(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def _rot_about(axis, angle):
    """Rodrigues rotation matrix about a unit-ish axis by angle."""
    k = _unit(np.asarray(axis, float))
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


def _finger_pts(mcp_pos, bones, abd, mcp_f, pip_f, dip_f):
    """4 points MCP,PIP,DIP,TIP. Flexion about y (curl toward -x); abd about x."""
    z = np.array([0.0, 0.0, 1.0])
    Rabd = _rx(abd)
    l_prox, l_mid, l_dist = bones
    d_prox = Rabd @ _ry(mcp_f) @ z
    p_pip = mcp_pos + l_prox * d_prox
    d_mid = Rabd @ _ry(mcp_f + pip_f) @ z
    p_dip = p_pip + l_mid * d_mid
    d_dist = Rabd @ _ry(mcp_f + pip_f + dip_f) @ z
    p_tip = p_dip + l_dist * d_dist
    return [mcp_pos, p_pip, p_dip, p_tip]


# thumb rest metacarpal direction: forward(+z), radial(+y), in front of palm(+x).
# Chosen inside the L20 thumb reachable cone so synthetic targets are plausible.
THUMB_REST = _unit(np.array([0.55, 0.45, 0.70]))


def _thumb_pts(opp, abd, flex, ip_flex):
    """Thumb CMC,MCP,IP,TIP. opp swings the thumb across the palm (about z),
    abd spreads it (about x), flex curls the proximal/distal about the local
    flexion axis."""
    cmc = THUMB_CMC.copy()
    l_meta, l_prox, l_dist = THUMB_BONES
    # opposition rotates about the palm normal (x) -- preserves the metacarpal's
    # palm-normal component, matching the L20 cmc_yaw axis; abduction (z) spreads.
    Rbase = _rx(opp) @ _rz(-abd)
    d_meta = Rbase @ THUMB_REST
    mcp = cmc + l_meta * d_meta
    flex_axis = Rbase @ np.array([0.0, 1.0, 0.0])  # local flexion axis
    d_prox = _rot_about(flex_axis, flex) @ d_meta
    ip = mcp + l_prox * d_prox
    d_dist = _rot_about(flex_axis, flex + ip_flex) @ d_meta
    tip = ip + l_dist * d_dist
    return [cmc, mcp, ip, tip]


# pose -> per-finger (abd, mcp_flex, pip_flex, dip_flex); thumb special
_FINGER_POSES = {
    "flat": {f: (0.0, 0.0, 0.0, 0.0) for f in BONES},
    "fist": {f: (0.0, 1.0, 1.3, 0.7) for f in BONES},
    "pinch": {  # index curls to meet thumb; others mild
        "index": (0.05, 0.9, 0.7, 0.4),
        "middle": (0.0, 0.5, 0.6, 0.3),
        "ring": (0.0, 0.5, 0.6, 0.3),
        "little": (0.0, 0.5, 0.6, 0.3),
    },
    "point": {
        "index": (0.0, 0.0, 0.0, 0.0),
        "middle": (0.0, 1.1, 1.3, 0.7),
        "ring": (0.0, 1.1, 1.3, 0.7),
        "little": (0.0, 1.1, 1.3, 0.7),
    },
    "thumbs_up": {f: (0.0, 1.2, 1.3, 0.7) for f in BONES},
}
# thumb params per pose: (opp, abd, flex, ip_flex)
_THUMB_POSES = {
    "flat": (0.0, 0.10, 0.0, 0.0),
    "fist": (0.6, 0.0, 0.9, 0.9),
    "pinch": (0.7, 0.0, 0.7, 0.6),
    "point": (0.5, 0.0, 0.7, 0.7),
    "thumbs_up": (-0.3, 0.3, -0.2, 0.0),  # extended outward/up (extreme)
}


def make_landmarks(pose: str, side: str = "right"):
    """Return a (21,3) numpy array of landmarks for the named pose/side."""
    pts = [None] * 21
    pts[WRIST] = np.array([0.0, 0.0, 0.0])
    for f in ("index", "middle", "ring", "little"):
        mcp = np.array([0.0, MCP_Y[f], PALM_LEN])
        abd, mf, pf, df = _FINGER_POSES[pose][f]
        a, b, c, d = GROUPS[f]
        fp = _finger_pts(mcp, BONES[f], abd, mf, pf, df)
        for idx, p in zip((a, b, c, d), fp):
            pts[idx] = p
    tp = _thumb_pts(*_THUMB_POSES[pose])
    for idx, p in zip(GROUPS["thumb"], tp):
        pts[idx] = p
    arr = np.array(pts, dtype=float)
    if side == "left":
        arr = arr.copy()
        arr[:, 1] *= -1.0  # mirror radial axis -> left hand chirality
    return arr


POSES = list(_THUMB_POSES.keys())


def write_fixtures(out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    written = []
    for pose in POSES:
        for side in ("right", "left"):
            arr = make_landmarks(pose, side)
            obj = {
                "side": side,
                "frame": "hand_base",
                "t": 0.0,
                "landmarks": [[round(float(v), 6) for v in p] for p in arr],
                "_pose": pose,
            }
            fname = f"{pose}_{side}.json"
            with open(os.path.join(out_dir, fname), "w") as fh:
                json.dump(obj, fh, indent=2)
            written.append(fname)
    return written


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    fixtures = os.path.normpath(
        os.path.join(here, "..", "..", "tests", "g0_unit", "fixtures")
    )
    names = write_fixtures(fixtures)
    print(f"wrote {len(names)} fixtures to {fixtures}:")
    for n in names:
        print("  ", n)
