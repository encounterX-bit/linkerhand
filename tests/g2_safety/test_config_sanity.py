"""Test 8: force/current cap present and bounded; watchdog safe-pose in-limits."""
import numpy as np
import pytest

from src.safety import DEFAULT_CONFIG, ForceClampSpec, WatchdogSpec
from src.safety.config import HW_FORCE_HARD_MAX_N
from src.safety import SafetyFilter
from helpers import SIDES


def test_force_clamp_present_and_far_below_max():
    spec = DEFAULT_CONFIG.force
    assert spec.max_grip_force_N > 0.0
    # FAR below the 100 N grip max, and within the policy fraction:
    assert spec.max_grip_force_N <= spec.max_fraction_of_hw_max * HW_FORCE_HARD_MAX_N
    assert spec.max_fraction_of_hw_max <= 0.5
    assert spec.per_joint_current_a > 0.0
    assert spec.is_sane()


def test_force_spec_requires_hw_token():
    # comms must refuse to actuate unless a HUMAN set HW_ENABLE_TOKEN.
    assert DEFAULT_CONFIG.force.require_hw_enable_token is True


def test_insane_force_spec_detected():
    bad = ForceClampSpec(max_grip_force_N=90.0)   # ~max, not "far below"
    assert bad.is_sane() is False


@pytest.mark.parametrize("side", SIDES)
def test_watchdog_safe_pose_in_limits(side):
    wd = DEFAULT_CONFIG.watchdog
    pose = wd.safe_pose_list()
    assert len(pose) == 20
    assert wd.stale_timeout_s > 0.0
    f = SafetyFilter(side)
    lims = f.model.fk.active_limits()
    for idx, (lo, hi) in lims.items():
        assert lo - 1e-9 <= pose[idx] <= hi + 1e-9
    for i in (11, 12, 13, 14):
        assert pose[i] == 0.0
    # the safe pose is a fixed point of the filter (already safe + open)
    r = f.filter(pose, pose, 0.033)
    assert r["modified"] is False


def test_safe_pose_is_collision_free():
    f = SafetyFilter("right")
    assert f.model.max_penetration(DEFAULT_CONFIG.watchdog.safe_pose_list(), 0.0) == 0.0
