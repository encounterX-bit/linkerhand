"""Test 4: input teleport -> output respects max joint velocity given dt."""
import numpy as np
import pytest

from helpers import SIDES, rand_in_limits


@pytest.mark.parametrize("side", SIDES)
def test_teleport_bounded_by_vmax_dt(side, request):
    f = request.getfixturevalue("filt_right" if side == "right" else "filt_left")
    vmax = f.cfg.max_joint_vel_rad_s
    rng = np.random.default_rng(21)
    for dt in (0.001, 0.01, 0.0333):
        for _ in range(30):
            prev = rand_in_limits(f.model, rng)
            teleport = rand_in_limits(f.model, rng)
            r = f.filter(teleport, prev, dt)
            out = np.array(r["joint_rad"])
            step = np.abs(out - np.array(prev))
            assert np.all(step <= vmax * dt + 1e-9), \
                f"joint moved {step.max():.4f} > {vmax*dt:.4f} in one frame"


def test_teleport_flags_rate_limit_reason(filt_right):
    prev = [0.0] * 20
    teleport = [0.0] * 20
    teleport[1] = 1.4          # full index flex in one frame
    r = filt_right.filter(teleport, prev, 0.001)
    assert "rate_limit" in (r["reason"] or "")
    assert r["modified"] is True
    assert abs(r["joint_rad"][1] - 0.0) <= filt_right.cfg.max_joint_vel_rad_s * 0.001 + 1e-9


def test_small_motion_within_band_not_rate_limited(filt_right):
    prev = [0.0] * 20
    cand = [0.0] * 20
    cand[1] = 0.001            # tiny, well inside vmax*dt
    r = filt_right.filter(cand, prev, 0.0333)
    assert "rate_limit" not in (r["reason"] or "")


def test_dt_zero_or_none_disables_rate_limit(filt_right):
    # With dt<=0 we cannot form a velocity band; rate limiting is disabled so a
    # (still limit/collision-checked) candidate is not spuriously frozen.
    cand = [0.0] * 20
    cand[1] = 1.0
    r = filt_right.filter(cand, [0.0] * 20, 0.0)
    assert "rate_limit" not in (r["reason"] or "")
