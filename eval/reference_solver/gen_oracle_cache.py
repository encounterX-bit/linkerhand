"""Generate cached G0 test fixtures (run offline; the oracle is slow).

Writes tests/g0_unit/fixtures/oracle_cache.json with two pose sets:

  reachable : N landmark sets synthesised from random in-limit robot configs
              (round-trip). The target directions are exactly reachable, so the
              closed-form must achieve J ~ 0 (>= the oracle's 0). No oracle run
              needed for these -- the bound is J_cf <= eps.
  plausible : M landmark sets from a random HUMAN-hand model, WITH full oracle
              solutions + J_oracle. The closed-form must satisfy
              J_cf <= J_oracle + eps (match-or-beat the slow optimiser), incl.
              genuinely under-actuated (nearest-reachable) targets.

Determinism: fixed RNG seeds; do not call Date/random at import.
"""
from __future__ import annotations

import json
import os

import numpy as np

from .model import L20Model, FINGERS
from .oracle import solve_oracle
from .objective import objective_J
from .landmarks import finger_segment_dirs
from . import synth_landmarks as S

N_REACHABLE = 1000
M_PLAUSIBLE = 80


def _synth_from_config(cfg, model):
    """21x3 landmarks whose per-finger u_prox/u_dist == the robot's r_prox/r_dist
    for `cfg` (a {semantic_idx: angle} map). Arbitrary per-finger anchor points
    and bone lengths -> exercises scale/translation invariance too."""
    lm = np.zeros((21, 3))
    for spec in FINGERS.values():
        jv = {spec.idx_to_joint()[i]: cfg.get(i, 0.0) for i in spec.idx_to_joint()}
        rp, rd = model.segment_dirs(spec, jv)
        a, b, c, d = spec.landmarks
        lm[a] = np.array([0.013 * a + 0.05, 0.2, 0.3])
        lm[b] = lm[a] + 0.042 * rp
        lm[d] = lm[b] + 0.031 * rd
        lm[c] = lm[b] + 0.015 * rd  # collapsed anchor (not used by either side)
    return lm


def total_J(lm, joint_rad, model):
    tot = 0.0
    for spec in FINGERS.values():
        up, ud = finger_segment_dirs(lm, spec.name)
        jv = {spec.idx_to_joint()[i]: joint_rad[i] for i in spec.idx_to_joint()}
        rp, rd = model.segment_dirs(spec, jv)
        tot += objective_J(rp, up, rd, ud)
    return tot


def build(side="right"):
    model = L20Model(side)
    reachable = []
    rng = np.random.default_rng(2024)
    for _ in range(N_REACHABLE):
        cfg = {i: rng.uniform(lo, hi)
               for spec in FINGERS.values()
               for i, (lo, hi) in model.finger_limits(spec).items()}
        lm = _synth_from_config(cfg, model)
        reachable.append({"landmarks": [[float(v) for v in p] for p in lm]})

    plausible = []
    rng = np.random.default_rng(99)
    for k in range(M_PLAUSIBLE):
        fp = {f: (rng.uniform(-0.15, 0.15), rng.uniform(0, 1.2),
                  rng.uniform(0, 1.4), rng.uniform(0, 0.9)) for f in S.BONES}
        S._FINGER_POSES["_g"] = fp
        S._THUMB_POSES["_g"] = (rng.uniform(-0.3, 1.0), rng.uniform(0, 0.4),
                                rng.uniform(-0.3, 0.9), rng.uniform(0, 0.9))
        lm = S.make_landmarks("_g", side)
        out = solve_oracle(lm.tolist(), side, model=model)
        plausible.append({
            "landmarks": [[float(v) for v in p] for p in lm],
            "oracle_joint_rad": out["joint_rad"],
            "J_oracle": total_J(lm, out["joint_rad"], model),
        })
    return {"side": side, "reachable": reachable, "plausible": plausible}


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    fixtures = os.path.normpath(os.path.join(here, "..", "..",
                                             "tests", "g0_unit", "fixtures"))
    for side in ("right", "left"):
        data = build(side)
        path = os.path.join(fixtures, f"oracle_cache_{side}.json")
        with open(path, "w") as fh:
            json.dump(data, fh)
        print(f"wrote {path}: {len(data['reachable'])} reachable, "
              f"{len(data['plausible'])} plausible")
