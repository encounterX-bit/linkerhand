"""G1 test 3 — limits, reserved DoF, mimic ranges, no-NaN.

Every config the harness ever commands — from G0 fixtures, random landmark clouds,
and round-trip configs — must stay inside the real URDF joint ranges, keep the
reserved DoF (idx 11-14) at exactly 0, keep the manually-enforced mimics in their
dependent ranges, and yield finite FK. The solver clamps; this asserts the property
holds at the sim boundary where the joints are actually placed.
"""
import glob
import json
import os

import numpy as np
import pytest

from src.sim import ACTIVE_IDX, RESERVED_IDX, FINGER_ORDER
from src.sim.pipeline import track_frame
from src.sim.synth import random_config, config_to_landmarks

HERE = os.path.dirname(os.path.abspath(__file__))
G0_FIXTURES = os.path.join(os.path.dirname(HERE), "g0_unit", "fixtures")
EPS = 1e-6


def _fixture_landmarks(side):
    out = []
    for path in sorted(glob.glob(os.path.join(G0_FIXTURES, f"*_{side}.json"))):
        if "oracle_cache" in os.path.basename(path):
            continue
        out.append(json.load(open(path))["landmarks"])
    return out


def _random_landmark_clouds(kin, n, seed):
    """Plausible-ish clouds: random in-limits configs -> their FK landmarks, plus
    jittered variants to push the solver off the reachable manifold."""
    rng = np.random.default_rng(seed)
    clouds = []
    for _ in range(n):
        q = random_config(rng, kin.active_limits())
        lm = config_to_landmarks(kin, q)
        lm = np.asarray(lm) + rng.normal(0, 0.004, size=(21, 3))  # off-manifold jitter
        clouds.append(lm.tolist())
    return clouds


@pytest.mark.parametrize("kin", ["right", "left"], indirect=True)
def test_commands_within_limits_and_reserved_zero(kin):
    limits = kin.active_limits()
    mimic_limits = kin.mimic_limits()
    clouds = _fixture_landmarks(kin.side) + _random_landmark_clouds(kin, 150, seed=7)
    assert clouds, "no input clouds assembled"

    for lm in clouds:
        rec = track_frame(kin, lm, kin.side)
        tgt = rec["targets"]

        # contract: active_idx + clamped flag
        assert tgt["active_idx"] == ACTIVE_IDX
        assert tgt["clamped"] is True
        jr = tgt["joint_rad"]
        assert len(jr) == 20

        # reserved DoF are exactly zero
        for idx in RESERVED_IDX:
            assert jr[idx] == 0.0

        # every active joint within URDF limits
        for idx in ACTIVE_IDX:
            lo, hi = limits[idx]
            assert lo - EPS <= jr[idx] <= hi + EPS, (
                f"[{kin.side}] idx {idx}={jr[idx]} out of [{lo},{hi}]")

        # manually-enforced mimics within their dependent ranges
        ang = rec["applied"]
        for mname, (lo, hi) in mimic_limits.items():
            assert lo - EPS <= ang[mname] <= hi + EPS, (
                f"[{kin.side}] mimic {mname}={ang[mname]} out of [{lo},{hi}]")

        # FK finite everywhere
        for f in FINGER_ORDER:
            for v in rec["robot"][f]:
                assert np.all(np.isfinite(v))
            for e in rec["err"][f]:
                assert np.isfinite(e)
