"""FK authority agreement (kinematics-agent-refactor invariant).

``src/kinematics.L20FK`` is now the single FK authority (pure yourdfpy/analytic).
PyBullet stays in ``src/sim`` for dynamics only, but its KINEMATIC FK must still
agree with the authority to ~1e-8 (the historical oracle<->sim agreement), so the
sim's measurement and the solver/oracle share one geometry. This pins that down on
raw link transforms (origins + orientations), independent of the segment
definition.
"""
import numpy as np
import pytest

from src.kinematics import L20FK, FINGERS
from src.sim import L20Kinematics

SIDES = ("right", "left")
# every link that participates in a segment direction, both endpoints + distal.
_LINKS = sorted({l for spec in FINGERS.values()
                 for l in (spec.link_a, spec.link_b, spec.link_c)})


def _random_cfg(fk, rng):
    jv = {}
    for spec in FINGERS.values():
        for idx, (lo, hi) in fk.finger_limits(spec).items():
            jv[spec.idx_to_joint()[idx]] = float(rng.uniform(lo, hi))
    return jv


@pytest.mark.parametrize("side", SIDES)
def test_pure_fk_matches_pybullet(side):
    fk = L20FK(side)
    kin = L20Kinematics(side)
    try:
        rng = np.random.default_rng(2026)
        worst_pos = worst_rot = 0.0
        for _ in range(120):
            jv = _random_cfg(fk, rng)
            fk.set_cfg(jv)
            # build the 20-vector for PyBullet from the same per-joint values
            q = [0.0] * 20
            for spec in FINGERS.values():
                for idx, jn in spec.idx_to_joint().items():
                    q[idx] = jv[jn]
            kin.set_config(q)
            for link in _LINKS:
                p_fk = fk.link_origin(link)
                T = fk.transform(link)
                p_pb, R_pb = kin.link_frame(link)
                worst_pos = max(worst_pos, float(np.linalg.norm(p_fk - p_pb)))
                worst_rot = max(worst_rot,
                                float(np.linalg.norm(T[:3, :3] - R_pb)))
        assert worst_pos < 1e-7, f"[{side}] FK origin disagreement {worst_pos:.2e}"
        assert worst_rot < 1e-6, f"[{side}] FK orientation disagreement {worst_rot:.2e}"
    finally:
        kin.close()
