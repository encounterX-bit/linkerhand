"""G1 test 2 — manual mimic enforcement.

PyBullet ignores URDF <mimic>. The harness must set each coupled distal joint to
``ratio * driver`` by hand. This test pins down three things:

  (a) the ratios baked into the URDF match hardware/LIMITS.md (0.8917 non-thumb,
      1.1619 thumb),
  (b) after :meth:`set_config`, the mimic joint angles equal ratio * driver, and
  (c) the enforcement actually changes FK — the coupled distal LINK ORIENTATION
      (hence the physical fingertip) moves when the mimic is applied. (Note: the
      ADR-0003 segment *origins* are mimic-independent by construction, so this must
      be checked on link orientation / an off-origin point, not on segment_dirs.)

If the harness silently dropped the mimic, (b) and (c) would both fail — which is
exactly the "wrong FK, silently wrong everything" failure the ticket warns about.
"""
import numpy as np
import pytest

import pybullet as pb

from src.sim import L20Kinematics, N_JOINTS

# expected (driver, ratio) per coupled distal joint, from hardware/LIMITS.md.
# The thumb coupled-distal joint name differs by side (thumb_dip right / thumb_ip
# left; same mimic) so it is resolved from the URDF, keyed by its thumb_mcp driver.
EXPECTED = {
    "index_dip": ("index_pip", 0.8917),
    "middle_dip": ("middle_pip", 0.8917),
    "ring_dip": ("ring_pip", 0.8917),
    "pinky_dip": ("pinky_pip", 0.8917),
}
THUMB_DRIVER = "thumb_mcp"
THUMB_RATIO = 1.1619


def _thumb_mimic(kin):
    """(mimic_name, driver, mult, offset) for the thumb coupled distal joint."""
    for mname, (driver, mult, off) in kin.mimics.items():
        if driver == THUMB_DRIVER:
            return mname, driver, mult, off
    raise AssertionError("no thumb coupled-distal mimic found in URDF")


def _q(idx_vals):
    q = [0.0] * N_JOINTS
    for i, v in idx_vals.items():
        q[i] = v
    return q


@pytest.mark.parametrize("side", ["right", "left"])
def test_urdf_ratios_match_limits_md(side):
    k = L20Kinematics(side)
    try:
        for mname, (driver, ratio) in {**EXPECTED}.items():
            d, mult, off = k.mimics[mname]
            assert d == driver
            assert off == 0.0
            assert mult == pytest.approx(ratio, abs=1e-4)
        _tname, tdriver, tmult, toff = _thumb_mimic(k)
        assert tdriver == THUMB_DRIVER and toff == 0.0
        assert tmult == pytest.approx(THUMB_RATIO, abs=1e-4)
    finally:
        k.close()


@pytest.mark.parametrize("side", ["right", "left"])
def test_mimic_angles_enforced(side):
    k = L20Kinematics(side)
    try:
        # idx 15 thumb tip, 16-19 finger tips -> drive all distal drivers
        q = _q({15: 0.8, 16: 1.2, 17: 1.0, 18: 0.9, 19: 1.1})
        k.set_config(q)
        ang = k.joint_angles()
        for mname, (driver, ratio) in EXPECTED.items():
            assert ang[mname] == pytest.approx(ratio * ang[driver], abs=1e-9)
            assert ang[mname] != pytest.approx(0.0)  # actually moved
        tname, tdriver, tmult, _ = _thumb_mimic(k)
        assert ang[tname] == pytest.approx(tmult * ang[tdriver], abs=1e-9)
    finally:
        k.close()


@pytest.mark.parametrize("side", ["right", "left"])
def test_mimic_changes_fk(side):
    """The coupled distal link's orientation must respond to the mimic."""
    k = L20Kinematics(side)
    try:
        q = _q({16: 1.3})  # index pip -> index_dip mimic = 0.8917*1.3
        # enforced
        k.set_config(q)
        _, R_on = k.link_frame("index_distal")
        # hand-defeat the mimic: re-zero just the dip joint, recompute
        pb.resetJointState(k.body, k.jidx["index_dip"], 0.0, physicsClientId=k.cid)
        _, R_off = k.link_frame("index_distal")
        # a point fixed on the distal link, offset off its origin, must move
        off = np.array([0.0, 0.0, 0.02])
        delta = np.linalg.norm(R_on @ off - R_off @ off)
        assert delta > 1e-3, (
            f"[{side}] distal link orientation insensitive to mimic "
            f"(delta={delta:.2e}) — mimic not enforced in FK")
    finally:
        k.close()
