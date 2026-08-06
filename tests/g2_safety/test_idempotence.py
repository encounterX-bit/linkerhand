"""Test 1: already-safe configs pass through unchanged within eps."""
import numpy as np
import pytest

from src.safety import DEFAULT_CONFIG
from helpers import SIDES, sample_safe


@pytest.mark.parametrize("side", SIDES)
def test_zero_pose_unchanged(side, request):
    f = request.getfixturevalue("filt_right" if side == "right" else "filt_left")
    r = f.filter([0.0] * 20, [0.0] * 20, 0.033)
    assert r["modified"] is False
    assert r["reason"] is None
    assert np.allclose(r["joint_rad"], [0.0] * 20, atol=0.0)


@pytest.mark.parametrize("side", SIDES)
def test_safe_random_configs_unchanged(side, request):
    f = request.getfixturevalue("filt_right" if side == "right" else "filt_left")
    rng = np.random.default_rng(7)
    margin = f.cfg.separation_margin_m
    safe = sample_safe(f.model, rng, 25, margin)
    assert len(safe) >= 10
    for c in safe:
        # prev == candidate so the rate band cannot move it; it is already in
        # limits, reserved-clean and collision-free at the margin.
        r = f.filter(c, c, 0.033)
        assert r["modified"] is False, f"safe config flagged modified: {r['reason']}"
        assert np.allclose(r["joint_rad"], c, atol=DEFAULT_CONFIG.eps_rad)


def test_idempotent_on_own_output(filt_right):
    """Filtering an output again is a no-op (fixed point)."""
    rng = np.random.default_rng(11)
    from helpers import sample_colliding
    cols = sample_colliding(filt_right.model, rng, 8)
    for c in cols:
        once = filt_right.filter(c, c, 1.0)["joint_rad"]
        twice = filt_right.filter(once, once, 1.0)
        assert twice["modified"] is False
        assert np.allclose(twice["joint_rad"], once, atol=1e-6)
