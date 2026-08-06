"""Test 3: adversarial self-collision set (paper-ablation analog).

A large batch of self-colliding scenarios across all three pair categories; the
filter must drive every one to non-penetration. Also includes OUT-OF-LIMITS and
reserved-dirty adversarial inputs (the filter is the last line of defense even
if the solver already clamps).
"""
import numpy as np
import pytest

from helpers import SIDES, sample_colliding, rand_in_limits


@pytest.mark.parametrize("side", SIDES)
def test_adversarial_batch_no_penetration(side, request):
    f = request.getfixturevalue("filt_right" if side == "right" else "filt_left")
    rng = np.random.default_rng(42)
    cols = sample_colliding(f.model, rng, 80, min_depth=0.001)
    assert len(cols) >= 50
    worst_in, worst_out = 0.0, 0.0
    for c in cols:
        worst_in = max(worst_in, f.model.max_penetration(c, 0.0))
        out = f.filter(c, c, 1.0)["joint_rad"]
        worst_out = max(worst_out, f.model.max_penetration(out, 0.0))
    assert worst_in > 0.005, "adversarial set was not actually colliding"
    assert worst_out <= 1e-4, f"residual penetration {worst_out}"


@pytest.mark.parametrize("side", SIDES)
def test_out_of_limits_input_clamped_and_safe(side, request):
    f = request.getfixturevalue("filt_right" if side == "right" else "filt_left")
    rng = np.random.default_rng(13)
    lims = f.model.fk.active_limits()
    for _ in range(40):
        # push every active joint WAY past its range, both directions
        c = [0.0] * 20
        for idx, (lo, hi) in lims.items():
            span = hi - lo
            c[idx] = (hi + 5 * span) if rng.random() < 0.5 else (lo - 5 * span)
        c[11] = 3.0; c[12] = -2.0; c[13] = 1.0; c[14] = 0.5   # dirty reserved
        r = f.filter(c, None, 1.0)
        out = np.array(r["joint_rad"])
        # within limits
        for idx, (lo, hi) in lims.items():
            assert lo - 1e-6 <= out[idx] <= hi + 1e-6
        # reserved zeroed
        assert np.all(out[[11, 12, 13, 14]] == 0.0)
        assert np.all(np.isfinite(out))
        assert "limits" in (r["reason"] or "")


def test_nan_input_does_not_emit_nan(filt_right):
    # A perception/solver glitch producing NaN must not propagate to a command.
    c = [0.0] * 20
    c[16] = float("nan")
    r = filt_right.filter(c, [0.0] * 20, 0.033)
    # NaN is outside any finite band/limit; clamp must sanitise it.
    assert np.all(np.isfinite(r["joint_rad"]))
