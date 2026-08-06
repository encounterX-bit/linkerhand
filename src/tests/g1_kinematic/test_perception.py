"""Perception gate tests (feeds G0/G1) for src.perception.

Covers the ticket's test list:
  - schema conformance (21 pts, hand_base frame, all finite)
  - frame-convention: known pose -> expected geometry (silent-rotation guard),
    incl. EXACT agreement with the solver/oracle G0 fixtures
  - smoothing reduces jitter (variance drop) within a lag bound
  - handedness mapping, both sides + mirrored/non-mirrored
  - rate/latency at camera rate; no-detection / low-confidence handled
  - runs on synthetic/recorded streams (no live camera)
"""
import glob
import json
import os
import time

import numpy as np
import pytest

from src.perception import (
    HandPipeline,
    OneEuroConfig,
    RawDetection,
    Recorder,
    ReplayHandSource,
    SyntheticHandSource,
    build_basis,
    to_hand_base,
    to_l20_side,
)
from src.perception.indices import (
    CANONICAL_FLAT_RIGHT,
    INDEX_MCP,
    MIDDLE_MCP,
    PINKY_MCP,
    WRIST,
)

REPO = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
G0_FIX = os.path.join(REPO, "tests", "g0_unit", "fixtures")
SCHEMA = os.path.join(REPO, "contracts", "hand_landmarks.schema.json")
SIDES = ("left", "right")


def _g0_fixtures():
    out = []
    for path in sorted(glob.glob(os.path.join(G0_FIX, "*.json"))):
        if "oracle_cache" in os.path.basename(path):
            continue
        with open(path) as fh:
            obj = json.load(fh)
        if "landmarks" in obj and "side" in obj:
            out.append((os.path.basename(path), obj))
    return out


def _rot(ax, ay, az):
    cx, sx = np.cos(ax), np.sin(ax)
    cy, sy = np.cos(ay), np.sin(ay)
    cz, sz = np.cos(az), np.sin(az)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def _assert_contract_valid(d):
    assert set(["side", "landmarks", "frame", "t"]).issubset(d)
    assert d["side"] in ("left", "right")
    assert d["frame"] == "hand_base"
    assert isinstance(d["t"], (int, float))
    lm = d["landmarks"]
    assert len(lm) == 21
    for p in lm:
        assert len(p) == 3
        assert all(np.isfinite(v) for v in p)


# --------------------------------------------------------------------------- #
# 1. schema conformance
# --------------------------------------------------------------------------- #
def test_schema_file_matches_emitted_keys():
    with open(SCHEMA) as fh:
        schema = json.load(fh)
    assert schema["properties"]["frame"]["const"] == "hand_base"
    assert schema["properties"]["landmarks"]["minItems"] == 21


@pytest.mark.parametrize("side", SIDES)
def test_emitted_frames_conform(side):
    src = SyntheticHandSource(n_frames=20, side=side, noise=0.002, seed=1)
    pipe = HandPipeline(src)
    n = 0
    for pf in pipe.run():
        d = pf.to_contract()
        _assert_contract_valid(d)
        assert d["side"] == side
        n += 1
    assert n >= 18  # nearly all frames emitted


# --------------------------------------------------------------------------- #
# 2. frame-convention: the silent-rotation guard
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name,obj", _g0_fixtures())
def test_frame_matches_solver_oracle_fixtures(name, obj):
    """The G0 fixtures already live in the oracle/solver hand_base frame. Running
    them through perception's transform must be (numerically) the identity --
    otherwise perception silently rotates every target vs the solver."""
    lm = np.asarray(obj["landmarks"], dtype=float)
    out = to_hand_base(lm, obj["side"])
    assert np.allclose(out, lm, atol=1e-6), f"{name}: frame mismatch"


@pytest.mark.parametrize("name,obj", _g0_fixtures())
def test_frame_is_rigid_recovery(name, obj):
    """Place a fixture in an arbitrary camera pose (rotate+translate+scale); the
    transform must recover the original hand_base coordinates up to scale."""
    lm = np.asarray(obj["landmarks"], dtype=float)
    R = _rot(0.6, -0.4, 1.1)
    t = np.array([0.3, -0.2, 1.5])
    scale = 2.5
    cam = (scale * lm) @ R.T + t
    out = to_hand_base(cam, obj["side"])
    # recovered up to the global scale the transform preserves
    assert np.allclose(out / scale, lm, atol=1e-6), f"{name}: not rigid-recovered"


@pytest.mark.parametrize("side", SIDES)
def test_known_pose_axis_geometry(side):
    """Canonical flat hand: +z toward fingers, +x palm-normal flat (~0),
    radial MCP ordering correct for the side."""
    base = CANONICAL_FLAT_RIGHT.copy()
    if side == "left":
        base[:, 1] *= -1.0
    out = to_hand_base(base, side)
    # fingers point +z
    assert out[MIDDLE_MCP][2] > 0.05
    # palm (wrist + MCPs) is flat in x
    palm = out[[WRIST, INDEX_MCP, MIDDLE_MCP, PINKY_MCP]]
    assert np.allclose(palm[:, 0], 0.0, atol=1e-6)
    # radial chirality: RIGHT hand has index on +y (thumb side); LEFT is the
    # mirror image (index on -y), matching the mirrored G0 fixtures.
    if side == "right":
        assert out[INDEX_MCP][1] > out[PINKY_MCP][1]
    else:
        assert out[INDEX_MCP][1] < out[PINKY_MCP][1]


def test_degenerate_palm_rejected():
    lm = CANONICAL_FLAT_RIGHT.copy()
    lm[[WRIST, INDEX_MCP, MIDDLE_MCP, 13, PINKY_MCP]] = 0.0  # collapse palm
    with pytest.raises(ValueError):
        build_basis(lm, "right")


# --------------------------------------------------------------------------- #
# 3. smoothing
# --------------------------------------------------------------------------- #
def test_smoothing_reduces_jitter():
    """On a still noisy hand, smoothing cuts landmark variance substantially."""
    n = 120
    noisy = SyntheticHandSource(
        n_frames=n, side="right", noise=0.004, z_noise_mult=3.0, seed=7
    )
    # capture the raw (post-transform, pre-smooth) and smoothed streams
    raw_pipe = HandPipeline(SyntheticHandSource(
        n_frames=n, side="right", noise=0.004, z_noise_mult=3.0, seed=7), smoothing=False)
    sm_pipe = HandPipeline(noisy, smoothing=True,
                           one_euro=OneEuroConfig(min_cutoff=0.8, beta=0.02))
    raw = np.array([pf.landmarks for pf in raw_pipe.run()])
    sm = np.array([pf.landmarks for pf in sm_pipe.run()])
    # ignore the very first frames (filter warm-up) and compare variance of the
    # detrended signal (motion is identical, so jitter dominates the residual).
    raw_v = np.var(raw[10:] - np.mean(raw[10:], axis=0), axis=0).mean()
    sm_v = np.var(sm[10:] - np.mean(sm[10:], axis=0), axis=0).mean()
    assert sm_v < 0.6 * raw_v, f"smoothing did not cut variance: {sm_v} vs {raw_v}"


def test_smoothing_lag_bounded_on_ramp():
    """A clean slow motion must be tracked with bounded lag (not frozen)."""
    n = 80
    clean = HandPipeline(
        SyntheticHandSource(n_frames=n, side="right", noise=0.0, seed=0),
        smoothing=True, one_euro=OneEuroConfig(min_cutoff=1.5, beta=0.05),
    )
    truth = HandPipeline(
        SyntheticHandSource(n_frames=n, side="right", noise=0.0, seed=0),
        smoothing=False,
    )
    sm = np.array([pf.landmarks for pf in clean.run()])
    tr = np.array([pf.landmarks for pf in truth.run()])
    # steady-state tracking error small relative to the motion amplitude
    err = np.linalg.norm(sm[-1] - tr[-1], axis=1).max()
    amp = np.linalg.norm(tr.max(axis=0) - tr.min(axis=0), axis=1).max()
    assert err < 0.15 * amp, f"smoothing lag too large: {err} vs amp {amp}"


def test_smoothing_off_is_passthrough():
    src = SyntheticHandSource(n_frames=5, side="right", noise=0.0, seed=0)
    raw_dets = list(SyntheticHandSource(n_frames=5, side="right", noise=0.0, seed=0))
    pipe = HandPipeline(src, smoothing=False)
    outs = list(pipe.run())
    expected = to_hand_base(raw_dets[2].landmarks, "right")
    assert np.allclose(outs[2].landmarks, expected, atol=1e-9)


def test_smoothing_flag_does_not_mutate_one_euro_config():
    cfg = OneEuroConfig(enabled=True)
    src = SyntheticHandSource(n_frames=1, side="right", noise=0.0, seed=0)
    HandPipeline(src, smoothing=False, one_euro=cfg)
    assert cfg.enabled is True


@pytest.mark.parametrize(
    "kwargs",
    (
        {"min_cutoff": 0.0},
        {"beta": -0.01},
        {"d_cutoff": 0.0},
    ),
)
def test_one_euro_config_rejects_invalid_gains(kwargs):
    with pytest.raises(ValueError):
        OneEuroConfig(**kwargs)


# --------------------------------------------------------------------------- #
# 4. handedness
# --------------------------------------------------------------------------- #
def test_handedness_mapping_table():
    # non-mirrored image: MediaPipe label is swapped to physical side
    assert to_l20_side("Left", image_mirrored=False) == "right"
    assert to_l20_side("Right", image_mirrored=False) == "left"
    # mirrored (selfie): label already matches physical side
    assert to_l20_side("Left", image_mirrored=True) == "left"
    assert to_l20_side("Right", image_mirrored=True) == "right"
    with pytest.raises(ValueError):
        to_l20_side("banana")


@pytest.mark.parametrize("side", SIDES)
def test_pipeline_resolves_side_both_hands(side):
    # SyntheticHandSource emits the swapped camera label; pipeline (non-mirrored)
    # must recover the physical side.
    src = SyntheticHandSource(n_frames=6, side=side, noise=0.0, seed=0)
    pipe = HandPipeline(src, image_mirrored=False)
    sides = {pf.side for pf in pipe.run()}
    assert sides == {side}


# --------------------------------------------------------------------------- #
# 5. robustness: no-detection / low-confidence / NaN
# --------------------------------------------------------------------------- #
def test_holds_last_good_on_dropout_and_lowconf():
    src = SyntheticHandSource(
        n_frames=20, side="right", noise=0.0, seed=0,
        dropout_frames=(5, 6), low_conf_frames=(10,),
    )
    pipe = HandPipeline(src, min_score=0.5)
    frames = list(pipe.run())
    assert len(frames) == 20  # nothing dropped from the output stream
    for i in (5, 6, 10):
        assert frames[i].held is True
        assert frames[i].detected is False
        assert "held" in frames[i].warnings
        assert np.all(np.isfinite(frames[i].landmarks))
    # held frame equals the last good landmarks
    assert np.allclose(frames[5].landmarks, frames[4].landmarks)


def test_no_emit_before_first_good_frame():
    dets = [RawDetection(t=0.0, ok=False), RawDetection(t=0.1, ok=False)]
    pipe = HandPipeline(ReplayHandSource(dets))
    assert list(pipe.run()) == []


def test_never_emits_nan():
    bad = RawDetection(
        t=0.0, ok=True, landmarks=np.full((21, 3), np.nan),
        handedness="Left", score=0.9,
    )
    good = RawDetection(
        t=0.033, ok=True, landmarks=CANONICAL_FLAT_RIGHT.copy(),
        handedness="Left", score=0.9,
    )
    pipe = HandPipeline(ReplayHandSource([bad, good]))
    outs = list(pipe.run())
    # first frame (NaN, no prior good) -> nothing; second -> finite output
    assert len(outs) == 1
    assert np.all(np.isfinite(outs[0].landmarks))


def test_depth_confidence_flagged():
    src = SyntheticHandSource(n_frames=3, side="right", depth_confidence=0.2, seed=0)
    pipe = HandPipeline(src, depth_warn=0.5)
    for pf in pipe.run():
        assert pf.depth_confidence == pytest.approx(0.2)
        assert "low_depth_confidence" in pf.warnings


# --------------------------------------------------------------------------- #
# 6. rate / latency
# --------------------------------------------------------------------------- #
def test_per_frame_latency_meets_camera_rate():
    src = SyntheticHandSource(n_frames=200, side="right", noise=0.003, seed=3)
    dets = list(src)
    pipe = HandPipeline(SyntheticHandSource(n_frames=1, side="right"))
    # time just the processing (transform + smooth), excluding generation
    t0 = time.perf_counter()
    for d in dets:
        pipe.process(d)
    dt = (time.perf_counter() - t0) / len(dets)
    assert dt < 1.0 / 60.0, f"per-frame {dt*1e3:.2f} ms exceeds 60 Hz budget"


# --------------------------------------------------------------------------- #
# 7. recorder round-trip
# --------------------------------------------------------------------------- #
def test_recorder_roundtrip(tmp_path):
    src = SyntheticHandSource(n_frames=15, side="left", noise=0.002, seed=2)
    pipe = HandPipeline(src)
    rec = Recorder(out_dir=str(tmp_path))
    for pf in pipe.run():
        rec.add(pf)
    path = rec.save("unit_left", source="synthetic")
    assert os.path.exists(path)
    obj = Recorder.load(path)
    assert obj["frame"] == "hand_base"
    assert obj["side"] == "left"
    assert obj["n_frames"] == len(obj["frames"]) >= 13
    for f in obj["frames"]:
        _assert_contract_valid(f)


def test_committed_real_fixture_exists_and_valid():
    """The synthetic 'real' fixture is committed for sim-agent's G1 replay."""
    real_dir = os.path.join(REPO, "tests", "g1_kinematic", "fixtures", "real")
    files = [
        f for f in glob.glob(os.path.join(real_dir, "*.json"))
    ]
    assert files, "no recorded sequences in fixtures/real/"
    for path in files:
        obj = Recorder.load(path)
        assert obj["frame"] == "hand_base"
        assert obj["frames"]
        for f in obj["frames"]:
            _assert_contract_valid(f)
