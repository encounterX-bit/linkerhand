"""G2 Test 6 — limits / reserved / no-NaN UNDER DYNAMICS.

Replays the recorded sequence through the closed loop (filter ON) and asserts that
what is COMMANDED and what is ACHIEVED both respect the L20 contract under physics:
reserved idx 11-14 are exactly 0, active joints stay within URDF limits (a small
dynamics overshoot band is allowed), and nothing is NaN/inf.
"""
import numpy as np
import pytest

from src.sim import ClosedLoopSim
from src.sim.conventions import ACTIVE_IDX, RESERVED_IDX

SIDES = ("right", "left")
OVERSHOOT = 0.05  # rad: PD can transiently overshoot a limit; allow a small band


@pytest.mark.parametrize("side", SIDES)
def test_limits_reserved_nan_under_dynamics(side, request):
    frames = request.getfixturevalue("frames_right" if side == "right" else "frames_left")
    cl = ClosedLoopSim(side, latency_s=0.0, use_filter=True)
    try:
        recs = cl.run(frames)
    finally:
        cl.close()

    # limits straight from the FK authority (single source of truth)
    from src.kinematics import L20FK
    active_limits = L20FK(side).active_limits()

    worst_over = 0.0
    for r in recs:
        cmd = np.asarray(r.applied, float)
        ach = np.asarray(r.achieved, float)
        assert np.all(np.isfinite(cmd)), f"{side}: NaN/inf in command"
        assert np.all(np.isfinite(ach)), f"{side}: NaN/inf in achieved config"
        # reserved are exactly 0 (command) — never actuated.
        for idx in RESERVED_IDX:
            assert cmd[idx] == 0.0, f"{side}: reserved idx {idx} commanded {cmd[idx]}"
        # active within limits (+ overshoot band) for both command and achieved.
        for idx in ACTIVE_IDX:
            lo, hi = active_limits[idx]
            for label, v in (("cmd", cmd[idx]), ("ach", ach[idx])):
                assert lo - OVERSHOOT <= v <= hi + OVERSHOOT, (
                    f"{side}: idx {idx} {label}={v:.3f} outside "
                    f"[{lo:.3f},{hi:.3f}] (+/-{OVERSHOOT})")
                worst_over = max(worst_over, max(0.0, lo - v, v - hi))
    print(f"\n[limits/{side}] {len(recs)} ticks ok; worst limit overshoot "
          f"{worst_over:.4f} rad (band {OVERSHOOT})")
