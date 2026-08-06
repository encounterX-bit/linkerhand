"""G1 test 1 — reachable set (hard pass).

Round-trip a real robot config through the full pipeline:
    q  --FK-->  segment dirs  --landmarks-->  solve  -->  q'  --FK-->  achieved dirs
Because the target directions come from an actual in-limits robot pose, they lie in
the solver's image, so the *achieved* segment directions must track the target to
within 0.01 rad. This isolates kinematic round-trip fidelity (distinct from the G0
oracle agreement) measured through PyBullet FK.
"""
import numpy as np
import pytest

from src.sim import FINGER_ORDER, geodesic_angle
from src.sim.pipeline import track_frame
from src.sim.synth import random_config, landmarks_from_dirs

TOL = 0.01      # rad, hard pass per ticket
N_CONFIGS = 400


@pytest.mark.parametrize("kin", ["right", "left"], indirect=True)
def test_reachable_round_trip(kin):
    rng = np.random.default_rng(20260608)
    limits = kin.active_limits()
    worst = 0.0
    worst_where = None
    for _ in range(N_CONFIGS):
        q = random_config(rng, limits)
        # target = the robot's own segment directions at this real config
        kin.set_config(q)
        target = kin.segment_dirs()
        landmarks = landmarks_from_dirs(target)

        rec = track_frame(kin, landmarks, kin.side)
        for f in FINGER_ORDER:
            for si in (0, 1):
                e = geodesic_angle(rec["robot"][f][si], target[f][si])
                if e > worst:
                    worst, worst_where = e, (f, ("prox", "dist")[si])
    assert worst <= TOL, (
        f"[{kin.side}] reachable round-trip exceeded {TOL} rad: "
        f"worst={worst:.5f} at {worst_where}")
