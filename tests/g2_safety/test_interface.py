"""The locked seam: filter(candidate, prev_safe, dt) -> dict shape."""
import numpy as np
import pytest

from src.safety import filter as safety_filter, get_filter
from helpers import SIDES, rand_in_limits


def test_return_shape_keys(filt_right):
    r = filt_right.filter([0.0] * 20, [0.0] * 20, 0.033)
    assert set(r) == {"joint_rad", "clamped", "modified", "reason"}
    assert r["clamped"] is True
    assert isinstance(r["modified"], bool)
    assert r["reason"] is None or isinstance(r["reason"], str)
    assert len(r["joint_rad"]) == 20
    assert all(isinstance(x, float) for x in r["joint_rad"])


def test_module_level_callable_default_side():
    r = safety_filter([0.0] * 20, [0.0] * 20, 0.033)
    assert r["clamped"] is True and len(r["joint_rad"]) == 20


def test_accepts_l20_targets_dict_and_infers_side():
    cand = {"side": "left", "joint_rad": [0.0] * 20,
            "active_idx": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 16, 17, 18, 19],
            "clamped": True, "t": 0.0}
    r = safety_filter(cand, {"side": "left", "joint_rad": [0.0] * 20}, 0.033)
    assert len(r["joint_rad"]) == 20


def test_prev_safe_none_allowed(filt_right):
    # No previous config -> rate limiting simply disabled (box == limits).
    r = filt_right.filter([0.0] * 20, None, 0.033)
    assert r["clamped"] is True


@pytest.mark.parametrize("side", SIDES)
def test_output_is_contract_shaped_joint_rad(side):
    f = get_filter(side)
    rng = np.random.default_rng(0)
    r = f.filter(rand_in_limits(f.model, rng), None, 1.0)
    jr = r["joint_rad"]
    assert len(jr) == 20
    for i in (11, 12, 13, 14):
        assert jr[i] == 0.0


def test_bad_length_rejected(filt_right):
    with pytest.raises(ValueError):
        filt_right.filter([0.0] * 16, None, 0.033)
