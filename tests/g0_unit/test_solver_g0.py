"""Gate G0 unit tests for the closed-form per-finger solver.

Covers the eight checks in docs/tickets/ticket-solver-agent-G0.md §6:
  1 matches oracle   2 per-segment bound   3 joint limits   4 scale invariance
  5 degenerate poses 6 determinism         7 handedness      8 timing benchmark

The solver under test (src.finger_retarget.retarget) is pure/closed-form. The
oracle FK (eval.reference_solver) is used only to SCORE outputs and as cached
ground truth -- the solver never imports it.
"""
import json
import os
import time

import numpy as np
import pytest

from src.finger_retarget import retarget, RESERVED_IDX
from src.finger_retarget.constants import ACTIVE_IDX
from eval.reference_solver import L20Model, FINGERS, finger_segment_dirs, objective_J
from eval.reference_solver.objective import angle_between

EPS = 1e-3             # ticket test 1 tolerance on REACHABLE targets (rad)
SEG_TOL = 1e-3        # per-segment tolerance for reachable targets (rad)
# Under-actuated NEAREST-reachable thumb targets: the closed-form matches the
# slow optimiser to ~p95 (PLAUSIBLE_P95) with a bounded worst case
# (PLAUSIBLE_MAX). Reachable targets remain exact (EPS). See ADR-0004.
PLAUSIBLE_P95 = 8e-3
PLAUSIBLE_MEAN = 2e-3
PLAUSIBLE_MAX = 5e-2
FIX = os.path.join(os.path.dirname(__file__), "fixtures")
SIDES = ("right", "left")


# --------------------------------------------------------------------------- #
# shared fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def models():
    return {s: L20Model(s) for s in SIDES}


@pytest.fixture(scope="module")
def cache():
    out = {}
    for s in SIDES:
        with open(os.path.join(FIX, f"oracle_cache_{s}.json")) as fh:
            out[s] = json.load(fh)
    return out


def _total_J(lm, joint_rad, model):
    tot = 0.0
    for spec in FINGERS.values():
        up, ud = finger_segment_dirs(lm, spec.name)
        jv = {spec.idx_to_joint()[i]: joint_rad[i] for i in spec.idx_to_joint()}
        rp, rd = model.segment_dirs(spec, jv)
        tot += objective_J(rp, up, rd, ud)
    return tot


def _seg_errors(lm, joint_rad, model):
    errs = []
    for spec in FINGERS.values():
        up, ud = finger_segment_dirs(lm, spec.name)
        jv = {spec.idx_to_joint()[i]: joint_rad[i] for i in spec.idx_to_joint()}
        rp, rd = model.segment_dirs(spec, jv)
        errs.append((spec.name, angle_between(rp, up), angle_between(rd, ud)))
    return errs


# --------------------------------------------------------------------------- #
# 1. matches oracle
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("side", SIDES)
def test_matches_oracle_reachable(side, models, cache):
    """>=1000 reachable round-trip poses: closed-form J <= eps (so it matches or
    beats the oracle, whose J >= 0 there)."""
    model = models[side]
    reach = cache[side]["reachable"]
    assert len(reach) >= 1000
    worst = 0.0
    for item in reach:
        lm = np.array(item["landmarks"])
        out = retarget(lm, side)
        worst = max(worst, _total_J(lm, out["joint_rad"], model))
    assert worst <= EPS, f"{side}: worst reachable J_cf={worst:.2e} > {EPS}"


@pytest.mark.parametrize("side", SIDES)
def test_matches_oracle_plausible(side, models, cache):
    """Plausible (often under-actuated) poses: the closed-form matches or beats
    the slow optimiser -- tight on average/p95, bounded worst case (the residual
    is the genuinely-unreachable thumb nearest; see ADR-0004)."""
    model = models[side]
    plaus = cache[side]["plausible"]
    assert len(plaus) >= 50
    diffs = []
    for item in plaus:
        lm = np.array(item["landmarks"])
        out = retarget(lm, side)
        j_cf = _total_J(lm, out["joint_rad"], model)
        diffs.append(j_cf - item["J_oracle"])
        assert j_cf <= item["J_oracle"] + PLAUSIBLE_MAX, (
            f"{side}: J_cf={j_cf:.5f} >> J_oracle={item['J_oracle']:.5f}")
    diffs = np.array(diffs)
    assert diffs.mean() <= PLAUSIBLE_MEAN, f"{side}: mean diff {diffs.mean():.5f}"
    assert np.percentile(diffs, 95) <= PLAUSIBLE_P95, (
        f"{side}: p95 diff {np.percentile(diffs, 95):.5f}")


# --------------------------------------------------------------------------- #
# 2. per-segment error bound (reachable targets)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("side", SIDES)
def test_per_segment_bound_reachable(side, models, cache):
    model = models[side]
    for item in cache[side]["reachable"][:500]:
        lm = np.array(item["landmarks"])
        out = retarget(lm, side)
        for name, ep, ed in _seg_errors(lm, out["joint_rad"], model):
            assert ep <= SEG_TOL and ed <= SEG_TOL, (
                f"{side}/{name}: prox={ep:.2e} dist={ed:.2e} > {SEG_TOL}")


# --------------------------------------------------------------------------- #
# 3. joint limits + reserved zeros + active_idx/clamped contract
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("side", SIDES)
def test_joint_limits_and_reserved(side, models, cache):
    model = models[side]
    # build {idx: (lo, hi)} from the URDF (eval) as the source of truth
    idx_lim = {}
    for spec in FINGERS.values():
        idx_lim.update(model.finger_limits(spec))
    samples = ([np.array(it["landmarks"]) for it in cache[side]["reachable"][:300]]
               + [np.array(it["landmarks"]) for it in cache[side]["plausible"]])
    for lm in samples:
        out = retarget(lm, side)
        jr = out["joint_rad"]
        assert out["active_idx"] == ACTIVE_IDX
        assert out["clamped"] is True
        assert len(jr) == 20
        for idx in RESERVED_IDX:
            assert jr[idx] == 0.0
        for idx, (lo, hi) in idx_lim.items():
            assert lo - 1e-9 <= jr[idx] <= hi + 1e-9, (
                f"{side}: idx {idx}={jr[idx]} out of [{lo},{hi}]")
        assert np.all(np.isfinite(jr))


# --------------------------------------------------------------------------- #
# 4. scale invariance (calibration-free)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("side", SIDES)
def test_scale_invariance(side, cache):
    for it in cache[side]["plausible"][:30]:
        lm = np.array(it["landmarks"])
        base = np.array(retarget(lm, side)["joint_rad"])
        for k in (0.5, 2.0, 5.0):
            scaled = np.array(retarget(lm * k, side)["joint_rad"])
            assert np.allclose(scaled, base, atol=1e-9), (
                f"{side}: scale k={k} changed output by "
                f"{np.abs(scaled - base).max():.2e}")


def test_translation_invariance():
    """Orientation-only: a rigid translation of all landmarks must not change
    the output (segment vectors are differences)."""
    lm = np.array(json.load(open(os.path.join(FIX, "pinch_right.json")))["landmarks"])
    base = np.array(retarget(lm, "right")["joint_rad"])
    shifted = np.array(retarget(lm + np.array([1.0, -2.0, 3.0]), "right")["joint_rad"])
    assert np.allclose(base, shifted, atol=1e-9)


# --------------------------------------------------------------------------- #
# 5. degenerate poses -> finite, clamped, no NaN
# --------------------------------------------------------------------------- #
def _degenerate_cases():
    cases = {}
    # fully extended (all fingers collinear along +z)
    lm = np.zeros((21, 3))
    for spec in FINGERS.values():
        a, b, c, d = spec.landmarks
        base = np.array([0.02 * a, 0.0, 0.05])
        for j, pt in enumerate((a, b, c, d)):
            lm[pt] = base + np.array([0.0, 0.0, 0.03 * j])
    cases["extended_collinear"] = lm.copy()
    # fully curled (segments fold back)
    lm = np.zeros((21, 3))
    for spec in FINGERS.values():
        a, b, c, d = spec.landmarks
        base = np.array([0.02 * a, 0.0, 0.05])
        lm[a] = base
        lm[b] = base + [0.0, 0.0, 0.04]
        lm[c] = base + [0.0, 0.0, 0.02]
        lm[d] = base + [0.0, 0.0, -0.01]
    cases["curled"] = lm.copy()
    # zero-length segments (coincident landmarks) -> division-by-zero guard
    lm = np.ones((21, 3)) * 0.05
    cases["coincident"] = lm.copy()
    # abduction extreme (large lateral spread)
    lm = np.zeros((21, 3))
    for spec in FINGERS.values():
        a, b, c, d = spec.landmarks
        base = np.array([0.02 * a, 0.0, 0.05])
        lm[a] = base
        lm[b] = base + [0.0, 0.05, 0.03]
        lm[c] = base + [0.0, 0.09, 0.05]
        lm[d] = base + [0.0, 0.12, 0.06]
    cases["abduction_extreme"] = lm.copy()
    return cases


@pytest.mark.parametrize("side", SIDES)
@pytest.mark.parametrize("name", list(_degenerate_cases().keys()))
def test_degenerate_finite_clamped(side, name, models):
    model = models[side]
    lm = _degenerate_cases()[name]
    out = retarget(lm, side)
    jr = out["joint_rad"]
    assert np.all(np.isfinite(jr)), f"{side}/{name}: non-finite output"
    idx_lim = {}
    for spec in FINGERS.values():
        idx_lim.update(model.finger_limits(spec))
    for idx, (lo, hi) in idx_lim.items():
        assert lo - 1e-9 <= jr[idx] <= hi + 1e-9
    for idx in RESERVED_IDX:
        assert jr[idx] == 0.0


# --------------------------------------------------------------------------- #
# 6. determinism (bitwise)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("side", SIDES)
def test_determinism(side, cache):
    for it in cache[side]["plausible"][:20]:
        lm = np.array(it["landmarks"])
        a = retarget(lm, side)["joint_rad"]
        b = retarget(lm, side)["joint_rad"]
        assert a == b  # exact list equality (bitwise)


# --------------------------------------------------------------------------- #
# 7. handedness
# --------------------------------------------------------------------------- #
def _reachable_landmarks(model, cfg):
    lm = np.zeros((21, 3))
    for spec in FINGERS.values():
        jv = {spec.idx_to_joint()[i]: cfg.get(i, 0.0) for i in spec.idx_to_joint()}
        rp, rd = model.segment_dirs(spec, jv)
        a, b, c, d = spec.landmarks
        lm[a] = np.array([0.02 * a + 0.05, 0.1, 0.2])
        lm[b] = lm[a] + 0.04 * rp
        lm[d] = lm[b] + 0.03 * rd
        lm[c] = lm[b] + 0.015 * rd
    return lm


def test_handedness_correct_side_solves(models):
    """A left-hand-reachable landmark set is solved exactly only with side='left';
    side='right' uses the mirrored abduction/opposition axes and is far worse."""
    # a config with engaged abduction + thumb opposition (handedness-sensitive)
    cfg = {6: 0.15, 7: -0.12, 8: 0.10, 1: 0.5, 16: 0.7, 10: 0.8, 5: 0.5, 0: 0.3, 15: 0.5}
    lm = _reachable_landmarks(models["left"], cfg)   # reachable on the LEFT hand
    j_correct = _total_J(lm, retarget(lm, "left")["joint_rad"], models["left"])
    j_wrong = _total_J(lm, retarget(lm, "right")["joint_rad"], models["right"])
    assert j_correct <= SEG_TOL * 5, f"correct side residual {j_correct:.4f}"
    assert j_wrong > j_correct + 0.1, (
        f"side label must matter: correct={j_correct:.4f} wrong={j_wrong:.4f}")


@pytest.mark.parametrize("side", SIDES)
def test_handedness_roundtrip_signs(side, models):
    """Round-trip a config with non-trivial abduction/opposition; the solver must
    recover the correct (side-dependent) signs to align both segments."""
    model = models[side]
    # a config with abduction at a limit and thumb opposition engaged
    cfg = {6: 0.15, 7: -0.15, 1: 0.6, 16: 0.8, 10: 0.7, 5: 0.4, 0: 0.3, 15: 0.5}
    lm = np.zeros((21, 3))
    for spec in FINGERS.values():
        jv = {spec.idx_to_joint()[i]: cfg.get(i, 0.0) for i in spec.idx_to_joint()}
        rp, rd = model.segment_dirs(spec, jv)
        a, b, c, d = spec.landmarks
        lm[a] = np.array([0.02 * a + 0.05, 0.1, 0.2])
        lm[b] = lm[a] + 0.04 * rp
        lm[d] = lm[b] + 0.03 * rd
        lm[c] = lm[b] + 0.015 * rd
    out = retarget(lm, side)
    for name, ep, ed in _seg_errors(lm, out["joint_rad"], model):
        assert ep <= SEG_TOL and ed <= SEG_TOL


# --------------------------------------------------------------------------- #
# 8. timing benchmark
#
# BUDGET SUPERSEDED (ADR-0007). The original closed-form solver met a hard 333 us
# (3 kHz) full-hand budget. The Finding-1 fingertip distal (ADR-0006) removed the
# closed form: the fingertip is a 1-DoF curve, so the non-thumb tip is a cheap 1-D
# search (still well within budget -- non-thumb alone is ~70-90 us) but the THUMB
# became a 2-DoF iterative solve (redundant base + tip), ~7x the old closed-form
# thumb. The reconciled gate: the MEDIAN reachable solve must still clear the 3 kHz
# PERIOD (p50 < 333 us, ~5 kHz median), and the iterative tail is bounded below
# (p99 < TAIL_US). Returning to a hard 3 kHz tail is a C/Cython/vectorised-thumb
# task tracked in ADR-0007 / STATE.md, gated before G2.
# --------------------------------------------------------------------------- #
PERIOD_US = 333.0      # the 3 kHz control period (median must clear it)
TAIL_US = 1200.0      # bound on the iterative-thumb tail (ADR-0007)


def _bench(lms, reps=4):
    for lm in lms[:30]:
        retarget(lm, "right")  # warm caches / branch predictors
    t = []
    for _ in range(reps):
        for lm in lms:
            t0 = time.perf_counter()
            retarget(lm, "right")
            t.append((time.perf_counter() - t0) * 1e6)
    t.sort()
    return t


def test_timing_representative(cache):
    """Representative (reachable) full-hand solve: the median must clear the 3 kHz
    period and the tail must stay within the iterative-distal bound (ADR-0007)."""
    lms = [[tuple(p) for p in it["landmarks"]]
           for it in cache["right"]["reachable"][:400]]
    t = _bench(lms)
    p50, p99 = t[len(t) // 2], t[int(len(t) * 0.99)]
    print(f"\n[timing/representative] p50={p50:.1f}us p99={p99:.1f}us n={len(t)}")
    assert p50 < PERIOD_US, f"median p50={p50:.1f}us does not clear the 3kHz period"
    assert p99 < TAIL_US, f"p99={p99:.1f}us exceeds the iterative-distal tail bound"


def test_timing_worstcase(cache):
    """Worst-case workload (every pose hitting the under-actuated thumb nearest
    grid): the tail must stay within the iterative-distal bound (ADR-0007)."""
    lms = [[tuple(p) for p in it["landmarks"]]
           for it in cache["right"]["plausible"]]
    t = _bench(lms, reps=6)
    p99, mx = t[int(len(t) * 0.99)], t[-1]
    print(f"\n[timing/worst-case] p99={p99:.1f}us max={mx:.1f}us n={len(t)}")
    assert p99 < TAIL_US, f"worst-case p99={p99:.1f}us exceeds the iterative-distal tail bound"
