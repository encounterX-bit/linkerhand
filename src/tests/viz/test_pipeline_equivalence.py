"""Headless equivalence: the viz loop is wired correctly, not a drifting copy.

Replaying the committed ``synthetic_openclose`` through the viz loop must produce
the SAME joint trajectory the existing pipeline produces on identical landmarks:

  * filter OFF -> equals ``finger_retarget.retarget()`` per frame (the G1
    track_frame candidate), and
  * filter ON  -> equals ``retarget() -> safety.filter()`` with prev_safe
    threaded + dt = 1/30 (exactly what the G2 closed-loop commands).

Plus: the PyBullet kinematic set (DIRECT, headless) actually applies the
commanded active joints. All headless (DIRECT) — no camera, no GUI.
"""
import numpy as np
import pytest

from src.finger_retarget import retarget
from src import safety
from src.viz import replay_stream, drive, L20VizModel, DEFAULT_DT
from src.viz.core import teleop_command

SIDES = ("right", "left")


def _retarget_reference(side):
    return [retarget(lm, side=side)["joint_rad"] for _s, lm, _t in replay_stream(side)]


def _filter_reference(side, dt=DEFAULT_DT):
    """Independently reconstruct the retarget->filter command sequence, threading
    prev_safe exactly as the G2 closed-loop does."""
    prev_safe = None
    cmds = []
    for _s, lm, _t in replay_stream(side):
        cand = retarget(lm, side=side)
        safe = safety.filter(cand, prev_safe, dt, side=side)
        cmds.append(list(safe["joint_rad"]))
        prev_safe = safe
    return cmds


@pytest.mark.parametrize("side", SIDES)
def test_fixture_loads(side):
    frames = list(replay_stream(side))
    assert len(frames) == 90
    s, lm, t = frames[0]
    assert s == side
    assert lm.shape == (21, 3)
    assert np.all(np.isfinite(lm))


@pytest.mark.parametrize("side", SIDES)
def test_viz_loop_no_filter_equals_retarget(side):
    """filter OFF: the viz loop reproduces the raw retarget trajectory exactly
    (same code path as the G1 pipeline's track_frame targets)."""
    recs = drive(None, replay_stream(side), use_filter=False)
    ref = _retarget_reference(side)
    assert len(recs) == len(ref) == 90
    for rec, exp in zip(recs, ref):
        assert rec["command"] == exp            # bit-exact, no drift


@pytest.mark.parametrize("side", SIDES)
def test_viz_loop_with_filter_equals_retarget_then_filter(side):
    """filter ON: the viz loop reproduces retarget->safety.filter exactly (the
    command the G2 closed-loop computes per tick)."""
    recs = drive(None, replay_stream(side), use_filter=True, dt=DEFAULT_DT)
    ref = _filter_reference(side, dt=DEFAULT_DT)
    assert len(recs) == len(ref) == 90
    for rec, exp in zip(recs, ref):
        assert np.array_equal(np.asarray(rec["command"]), np.asarray(exp))


@pytest.mark.parametrize("side", SIDES)
def test_reserved_always_zero(side):
    for use_filter in (False, True):
        recs = drive(None, replay_stream(side), use_filter=use_filter)
        for rec in recs:
            for idx in (11, 12, 13, 14):
                assert rec["command"][idx] == 0.0


@pytest.mark.parametrize("side", SIDES)
def test_headless_render_applies_commanded_joints(side):
    """The DIRECT PyBullet set applies exactly the commanded active joints
    (proves resetJointState wiring; reserved stay 0)."""
    model = L20VizModel(side, gui=False)
    try:
        recs = drive(model, replay_stream(side), use_filter=True)
        # re-apply the final command and read back the active joints
        last = recs[-1]["command"]
        model.set_joints(last)
        applied = model.applied_active()
        from src.kinematics import ACTIVE_IDX
        for idx in ACTIVE_IDX:
            assert applied[idx] == pytest.approx(last[idx], abs=1e-9)
        for idx in (11, 12, 13, 14):
            assert applied[idx] == 0.0
    finally:
        model.close()


def test_drive_resets_prev_safe_on_side_change():
    """A mixed-side stream must not feed a prev_safe from the other side into the
    per-side filter (which would raise). The loop resets on side change."""
    r = list(replay_stream("right"))[:5]
    l = list(replay_stream("left"))[:5]
    recs = drive(None, iter(r + l), use_filter=True)
    assert len(recs) == 10
    assert [x["side"] for x in recs[:5]] == ["right"] * 5
    assert [x["side"] for x in recs[5:]] == ["left"] * 5


def test_teleop_command_shape_and_keys():
    _s, lm, _t = next(replay_stream("right"))
    out = teleop_command(lm, "right")
    assert set(out) == {"command", "candidate", "safe", "modified", "reason"}
    assert len(out["command"]) == 20
    assert out["safe"] is not None
