"""Test 5: continuity / no chatter across a safe->colliding->safe sweep.

Stream a trajectory through a collision region with realistic per-frame dt and
rate limiting on (prev_safe = last output). The output must stay continuous
(bounded per-frame step) and must NOT oscillate at the collision boundary.
"""
import numpy as np
import pytest

from helpers import SIDES


def _sweep(f, a, b, n):
    """Filter a linear sweep a->b->a, threading prev_safe = last safe output."""
    a = np.array(a, float); b = np.array(b, float)
    pts = np.linspace(0, 1, n)
    traj = [a * (1 - t) + b * t for t in pts] + [a * t + b * (1 - t) for t in pts]
    dt = 1.0 / 30.0
    prev = list(a)
    outs = []
    for cand in traj:
        r = f.filter(list(cand), prev, dt)
        prev = r["joint_rad"]
        outs.append(np.array(prev))
    return outs


@pytest.mark.parametrize("side", SIDES)
def test_sweep_is_continuous(side, request):
    f = request.getfixturevalue("filt_right" if side == "right" else "filt_left")
    vmax = f.cfg.max_joint_vel_rad_s
    dt = 1.0 / 30.0
    # open hand -> a flexed/abducted pose that passes through a collision region
    a = [0.0] * 20
    b = [0.0] * 20
    b[1] = 1.2; b[2] = 1.2; b[16] = 1.4; b[17] = 1.4
    b[6] = 0.17; b[7] = -0.17
    outs = _sweep(f, a, b, 40)
    steps = [np.abs(outs[i + 1] - outs[i]).max() for i in range(len(outs) - 1)]
    # every frame respects the rate limit -> bounded, continuous
    assert max(steps) <= vmax * dt + 1e-9
    # output stays collision-free throughout the sweep
    for o in outs:
        assert f.model.max_penetration(o, 0.0) <= 1e-3


@pytest.mark.parametrize("side", SIDES)
def test_no_chatter_at_boundary(side, request):
    """Hold a candidate fixed right at the collision boundary; the output must
    settle (no limit-cycle oscillation frame to frame)."""
    f = request.getfixturevalue("filt_right" if side == "right" else "filt_left")
    cand = [0.0] * 20
    cand[1] = 0.9; cand[2] = 0.9; cand[16] = 1.1; cand[17] = 1.1
    cand[6] = 0.17; cand[7] = -0.17
    dt = 1.0 / 30.0
    prev = [0.0] * 20
    outs = []
    for _ in range(60):                 # hold the same candidate 60 frames
        r = f.filter(cand, prev, dt)
        prev = r["joint_rad"]
        outs.append(np.array(prev))
    tail = outs[-10:]
    # the last frames must be ~identical (settled), not oscillating
    spread = np.max([np.abs(tail[i] - tail[-1]).max() for i in range(len(tail))])
    assert spread <= 1e-4, f"output chattering at boundary, spread={spread}"
    assert f.model.max_penetration(outs[-1], 0.0) <= 1e-3
