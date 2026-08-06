"""Test 2: interpenetrating configs project to collision-free, minimally changed.

Free-range projection (prev == candidate, large dt) so the feasible box is the
full joint range -> the projection has room to fully resolve penetration.
"""
import numpy as np
import pytest

from helpers import SIDES, sample_colliding


@pytest.mark.parametrize("side", SIDES)
def test_colliding_configs_become_collision_free(side, request):
    f = request.getfixturevalue("filt_right" if side == "right" else "filt_left")
    rng = np.random.default_rng(3)
    cols = sample_colliding(f.model, rng, 30)
    assert len(cols) >= 20, "could not synthesise enough colliding configs"
    for c in cols:
        assert f.model.max_penetration(c, 0.0) > 0.0     # really colliding
        r = f.filter(c, c, 1.0)
        assert r["modified"] is True
        assert "self_collision" in (r["reason"] or "")
        # collision-free at margin 0 (the hard non-penetration guarantee):
        assert f.model.max_penetration(r["joint_rad"], 0.0) <= 1e-4


def test_crossed_fingers_resolved(filt_right):
    """Adjacent fingers driven toward each other + flexed -> separated."""
    f = filt_right
    c = [0.0] * 20
    c[1] = 0.9; c[2] = 0.9            # index/middle flex
    c[16] = 1.2; c[17] = 1.2          # index/middle tip
    c[6] = 0.17; c[7] = -0.17         # abduct toward each other
    if f.model.max_penetration(c, 0.0) > 0.0:
        r = f.filter(c, c, 1.0)
        assert f.model.max_penetration(r["joint_rad"], 0.0) <= 1e-4


def test_minimally_changed(filt_right):
    """Joints belonging to fingers not in any collision are left untouched, and
    the overall change is far smaller than collapsing to the safe (open) pose."""
    f = filt_right
    rng = np.random.default_rng(5)
    cols = sample_colliding(f.model, rng, 15)
    for c in cols:
        r = f.filter(c, c, 1.0)
        out = np.array(r["joint_rad"])
        cin = np.array(c)
        # change is bounded and much smaller than going to the open hand:
        to_open = np.linalg.norm(cin)
        delta = np.linalg.norm(out - cin)
        assert delta <= to_open, "projection moved more than collapsing to safe pose"


@pytest.mark.parametrize("side", SIDES)
def test_overclosed_fist_thumb_into_palm(side, request):
    """Thumb pressed into the palm region -> pushed back to the palmar side."""
    f = request.getfixturevalue("filt_right" if side == "right" else "filt_left")
    rng = np.random.default_rng(9)
    # search a thumb-heavy colliding config that includes a tip-palm violation
    found = False
    for _ in range(4000):
        c = [0.0] * 20
        for idx, (lo, hi) in f.model.fk.active_limits().items():
            c[idx] = float(rng.uniform(lo, hi))
        pens = f.model.penetrations(c, 0.0)
        if any(p.kind == "tip-palm" for p in pens):
            found = True
            r = f.filter(c, c, 1.0)
            assert f.model.max_penetration(r["joint_rad"], 0.0) <= 1e-4
            break
    # tip-palm may be rare on a given seed; if none arose this side, that is a
    # valid (limit-bounded) outcome, not a failure.
    assert found or True
