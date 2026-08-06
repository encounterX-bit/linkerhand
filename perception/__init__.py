"""src.perception -- vision -> hand_landmarks (ADR-0003 hand_base frame).

Public surface (coordinate with other modules via contracts only):
    HandSource, RawDetection, ReplayHandSource, SyntheticHandSource
    MediaPipeHandSource            (default real backend; lazy mediapipe import)
    to_hand_base, build_basis      (ADR-0003 frame transform)
    OneEuroConfig, LandmarkSmoother
    to_l20_side                    (handedness mapping)
    HandPipeline, ProcessedFrame
    Recorder
"""
from __future__ import annotations

from .frame import build_basis, to_hand_base
from .handedness import to_l20_side
from .indices import FINGER_LANDMARKS, N_LANDMARKS, PALM_LANDMARKS
from .one_euro import LandmarkSmoother, OneEuroConfig
from .pipeline import HandPipeline, ProcessedFrame
from .recorder import DEFAULT_REAL_DIR, Recorder
from .source import HandSource, RawDetection, ReplayHandSource, SyntheticHandSource

__all__ = [
    "HandSource",
    "RawDetection",
    "ReplayHandSource",
    "SyntheticHandSource",
    "MediaPipeHandSource",
    "to_hand_base",
    "build_basis",
    "OneEuroConfig",
    "LandmarkSmoother",
    "to_l20_side",
    "HandPipeline",
    "ProcessedFrame",
    "Recorder",
    "DEFAULT_REAL_DIR",
    "FINGER_LANDMARKS",
    "PALM_LANDMARKS",
    "N_LANDMARKS",
]


def __getattr__(name):
    # Lazy: importing mediapipe/opencv only when the real backend is used.
    if name == "MediaPipeHandSource":
        from .mediapipe_source import MediaPipeHandSource

        return MediaPipeHandSource
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
