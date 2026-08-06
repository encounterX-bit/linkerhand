"""VideoHandSource: a recorded clip carried through to a valid hand_landmarks.

All camera-free: fake cv2 capture + fake MediaPipe detector drive the REAL
inherited ``read()`` (the MediaPipeHandSource detection path), proving the source
reuses that path and differs only in frame acquisition + timing. We assert valid
contract emission via the EXISTING pipeline, LOW depth confidence (monocular),
graceful no-detection + EOF handling, and native-FPS timestamps. mediapipe /
opencv are never imported (the heavy deps live behind the parent constructor,
which these tests bypass with ``__new__``).
"""
import json
import os

import numpy as np
import pytest

from src.perception.video_source import (
    VideoHandSource, LOW_DEPTH_CONFIDENCE, resolve_fps,
)
from src.perception.indices import N_LANDMARKS, CANONICAL_FLAT_RIGHT

HERE = os.path.dirname(os.path.abspath(__file__))
CONTRACT = os.path.normpath(os.path.join(
    HERE, os.pardir, os.pardir, "contracts", "hand_landmarks.schema.json"))


# --- pure FPS resolution --------------------------------------------------- #
def test_resolve_fps_prefers_override():
    assert resolve_fps(30.0, override=60.0) == 60.0


def test_resolve_fps_uses_native_when_no_override():
    assert resolve_fps(24.0) == 24.0


def test_resolve_fps_rejects_zero_and_nonfinite_native():
    assert resolve_fps(0.0) == 30.0
    assert resolve_fps(float("nan")) == 30.0
    assert resolve_fps(float("inf")) == 30.0
    assert resolve_fps(-5.0) == 30.0


def test_resolve_fps_rejects_bad_override_falls_back_to_native():
    assert resolve_fps(24.0, override=0.0) == 24.0
    assert resolve_fps(24.0, override=float("nan")) == 24.0


# --- fakes that drive the inherited MediaPipe read() ----------------------- #
class _FakeLm:
    def __init__(self, xyz):
        self.x, self.y, self.z = float(xyz[0]), float(xyz[1]), float(xyz[2])


class _FakeWorld:
    def __init__(self, pts):
        self.landmark = [_FakeLm(p) for p in pts]


class _FakeClassification:
    def __init__(self, label, score):
        self.label, self.score = label, score


class _FakeHandedness:
    def __init__(self, label, score):
        self.classification = [_FakeClassification(label, score)]


class _FakeResult:
    """Mimics a mediapipe Hands.process() result. ``pts=None`` => no detection."""

    def __init__(self, pts=None, label="Left", score=0.95):
        if pts is None:
            self.multi_hand_world_landmarks = None
            self.multi_handedness = None
        else:
            self.multi_hand_world_landmarks = [_FakeWorld(pts)]
            self.multi_handedness = [_FakeHandedness(label, score)]


class _FakeHands:
    def __init__(self, results):
        self._results = list(results)
        self._i = 0

    def process(self, rgb):
        r = self._results[self._i]
        self._i += 1
        return r

    def close(self):
        pass


class _FakeCap:
    """``read()`` returns a frame per result, then ``(False, None)`` at EOF."""

    def __init__(self, n_frames):
        self._n = n_frames
        self._i = 0

    def read(self):
        if self._i >= self._n:
            return False, None
        self._i += 1
        return True, object()  # opaque BGR frame placeholder

    def release(self):
        pass


class _FakeCv2:
    COLOR_BGR2RGB = 4

    def cvtColor(self, frame, code):
        return frame


def _make_source(results, *, fps=24.0):
    """A VideoHandSource wired to fakes, WITHOUT the heavy-dep constructor."""
    src = VideoHandSource.__new__(VideoHandSource)
    src._cv2 = _FakeCv2()
    src._cap = _FakeCap(len(results))
    src._hands = _FakeHands(results)
    src.fps = fps
    src.native_fps = fps
    src.playback_rate = 1.0
    src._frame_idx = 0
    src.depth_confidence = LOW_DEPTH_CONFIDENCE
    src.path = "fake.mp4"
    return src


# --- read() behaviour ------------------------------------------------------ #
def test_detection_frame_is_low_confidence_and_well_formed():
    src = _make_source([_FakeResult(CANONICAL_FLAT_RIGHT)])
    det = src.read()
    assert det.ok
    assert det.depth_confidence == LOW_DEPTH_CONFIDENCE  # monocular => LOW
    assert det.landmarks.shape == (N_LANDMARKS, 3)
    assert np.all(np.isfinite(det.landmarks))
    assert det.handedness == "Left"


def test_no_detection_frame_is_not_ok_never_nan():
    src = _make_source([_FakeResult(pts=None)])
    det = src.read()
    assert det.ok is False
    assert det.landmarks is None  # pipeline holds last good; never a NaN cloud


def test_eof_raises_stopiteration():
    src = _make_source([_FakeResult(CANONICAL_FLAT_RIGHT)])
    src.read()
    with pytest.raises(StopIteration):
        src.read()


def test_honors_fps_in_timestamps():
    res = [_FakeResult(CANONICAL_FLAT_RIGHT) for _ in range(3)]
    src = _make_source(res, fps=25.0)
    ts = [src.read().t for _ in range(3)]
    assert ts == pytest.approx([0.0, 1.0 / 25.0, 2.0 / 25.0])


def test_frame_period_scales_with_playback_rate():
    src = _make_source([], fps=20.0)
    assert src.frame_period == pytest.approx(1.0 / 20.0)
    src.playback_rate = 2.0
    assert src.frame_period == pytest.approx(1.0 / 40.0)  # 2x faster -> shorter hold
    src.playback_rate = 0.0  # guarded -> treated as 1.0
    assert src.frame_period == pytest.approx(1.0 / 20.0)


# --- contract conformance via the existing pipeline ------------------------ #
def test_video_emits_valid_contract_low_confidence_warned():
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.load(open(CONTRACT))

    src = _make_source([_FakeResult(CANONICAL_FLAT_RIGHT, label="Left", score=0.97)])
    det = src.read()

    from src.perception.pipeline import HandPipeline
    from src.perception.source import ReplayHandSource
    pipe = HandPipeline(ReplayHandSource([det]), smoothing=False)
    out = list(pipe.run())
    assert len(out) == 1
    contract = out[0].to_contract()
    jsonschema.validate(contract, schema)
    assert contract["frame"] == "hand_base"
    assert contract["side"] == "right"            # swapped from camera-view "Left"
    assert contract["depth_confidence"] == LOW_DEPTH_CONFIDENCE
    # LOW monocular confidence -> the pipeline DOES raise the low-depth warning
    # (the opposite of the metric RealSense path).
    assert "low_depth_confidence" in contract["warnings"]


def test_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        VideoHandSource("/no/such/clip_xyz.mp4")
