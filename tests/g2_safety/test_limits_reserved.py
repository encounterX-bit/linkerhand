"""Test 6: output in-range; idx 11-14 == 0; mimic dependents in range; no NaN."""
import numpy as np
import pytest

from src.kinematics import L20FK
from helpers import SIDES, rand_in_limits, sample_colliding


@pytest.mark.parametrize("side", SIDES)
def test_output_in_limits_and_reserved_zero(side, request):
    f = request.getfixturevalue("filt_right" if side == "right" else "filt_left")
    rng = np.random.default_rng(31)
    lims = f.model.fk.active_limits()
    cases = [rand_in_limits(f.model, rng) for _ in range(40)]
    cases += sample_colliding(f.model, rng, 40)
    for c in cases:
        out = np.array(f.filter(c, None, 1.0)["joint_rad"])
        assert np.all(np.isfinite(out))
        assert np.all(out[[11, 12, 13, 14]] == 0.0)
        for idx, (lo, hi) in lims.items():
            assert lo - 1e-9 <= out[idx] <= hi + 1e-9


@pytest.mark.parametrize("side", SIDES)
def test_mimic_dependents_in_range(side, request):
    """Driver tip commands keep every mimic DEPENDENT within its URDF range."""
    f = request.getfixturevalue("filt_right" if side == "right" else "filt_left")
    fk = f.model.fk
    name_to_idx = f.model._name_to_idx
    rng = np.random.default_rng(32)
    cases = [rand_in_limits(f.model, rng) for _ in range(30)]
    cases += sample_colliding(f.model, rng, 20)
    for c in cases:
        out = f.filter(c, None, 1.0)["joint_rad"]
        for mj, (driver, mult, off) in fk.mimics.items():
            if driver not in name_to_idx or mj not in fk.limits:
                continue
            dep = mult * out[name_to_idx[driver]] + off
            lo, hi = fk.limits[mj]
            assert lo - 1e-6 <= dep <= hi + 1e-6, f"{mj} dependent {dep} out of [{lo},{hi}]"


def test_reserved_forced_even_if_input_nonzero(filt_right):
    c = [0.1] * 20
    out = filt_right.filter(c, None, 1.0)["joint_rad"]
    assert out[11] == out[12] == out[13] == out[14] == 0.0
