"""G1 test 4 — thumb axis (cmc_pitch/cmc_roll label) confirmation.

hardware/LIMITS.md flagged the thumb base (idx 0 = thumb_cmc_pitch) vs abduction
(idx 5 = thumb_cmc_roll) assignment as the one ambiguity. We confirm it
empirically against the kinematics: driving the BASE DoF must *flex the thumb
toward the palm* — an IN-palm-plane curl toward the fingers — whereas the
ABDUCTION DoF must lift the thumb OUT of the palm plane. If the two were swapped,
these signatures would swap too.

Palm frame (from the zero pose, no magic numbers):
  fwd    = mean finger proximal->middle direction (points distally, toward tips)
  width  = index_proximal - pinky_proximal (across the palm)
  normal = fwd x width  (out of the palm plane)

Flexion signature : |d_tip . fwd| dominates |d_tip . normal|, d_tip . fwd > 0.
Abduction signature: |d_tip . normal| dominates |d_tip . fwd|.
Frames are rendered to out/ for human review.
"""
import os

import numpy as np
import pytest

from src.sim import L20Kinematics, N_JOINTS
from src.sim import viz

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "out")

IDX_THUMB_BASE = 0   # thumb_cmc_pitch  (claimed flexion)
IDX_THUMB_ABD = 5    # thumb_cmc_roll   (claimed abduction)
N_STEPS = 12


def _palm_frame(kin):
    kin.set_config([0.0] * N_JOINTS)
    fdirs = []
    for f in ("index", "middle", "ring", "little"):
        rp, _ = kin.segment_dirs()[f]
        fdirs.append(rp)
    fwd = _unit(np.mean(fdirs, axis=0))
    width = _unit(kin.link_origin("index_proximal") - kin.link_origin("pinky_proximal"))
    normal = _unit(np.cross(fwd, width))
    return fwd, width, normal


def _sweep(kin, idx, lo_hi):
    """Drive joint `idx` from 0..max; return list of thumb-tip positions."""
    lo, hi = lo_hi
    tips = []
    for s in range(N_STEPS):
        q = [0.0] * N_JOINTS
        q[idx] = lo + (hi - lo) * s / (N_STEPS - 1)
        kin.set_config(q)
        tips.append(kin.link_origin("thumb_distal"))
    return tips


def _components(tips, basis):
    fwd, width, normal = basis
    d = tips[-1] - tips[0]
    return d @ fwd, d @ width, d @ normal


@pytest.mark.parametrize("side", ["right", "left"])
def test_thumb_base_is_flexion_not_abduction(side):
    k = L20Kinematics(side)
    try:
        basis = _palm_frame(k)
        limits = k.active_limits()

        base_tips = _sweep(k, IDX_THUMB_BASE, limits[IDX_THUMB_BASE])
        abd_tips = _sweep(k, IDX_THUMB_ABD, limits[IDX_THUMB_ABD])

        bf, bw, bn = _components(base_tips, basis)
        af, aw, an = _components(abd_tips, basis)

        # --- BASE (idx 0) flexes IN the palm plane, toward the fingers ---------
        assert bf > 0.0, f"[{side}] thumb base should curl toward fingers (fwd>0), got {bf:.3f}"
        assert abs(bf) > 3.0 * abs(bn), (
            f"[{side}] thumb base motion not predominantly in-plane flexion: "
            f"|fwd|={abs(bf):.3f} vs |normal|={abs(bn):.3f}")

        # --- ABDUCTION (idx 5) lifts OUT of the palm plane --------------------
        assert abs(an) > abs(af), (
            f"[{side}] thumb abduction not predominantly out-of-plane: "
            f"|normal|={abs(an):.3f} vs |fwd|={abs(af):.3f}")

        # --- the two are genuinely distinct (not the same DoF mislabeled) -----
        # base is far more in-plane than abduction is.
        assert abs(bn) / max(abs(bf), 1e-9) < abs(an) / max(abs(af), 1e-9)

        # human-review renders
        for name, tips in (("base_flexion_idx0", base_tips),
                           ("abduction_idx5", abd_tips)):
            comps = [(t @ basis[0], t @ basis[1], t @ basis[2]) for t in tips]
            comps = [(c[0] - comps[0][0], c[1] - comps[0][1], c[2] - comps[0][2])
                     for c in comps]
            viz.render_thumb_sweep(comps, basis, OUT_DIR, f"{side}_{name}",
                                   "fwd=toward-fingers, normal=out-of-palm")
    finally:
        k.close()


def _unit(v):
    v = np.asarray(v, dtype=float)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else np.array([0.0, 0.0, 1.0])
