"""src/viz — Stage 1 live RealSense -> sim L20 mirror (sim-only, post-G2).

A visualization layer on top of G2: it imports perception, ``finger_retarget``,
``safety``, ``src/kinematics`` and ``src/sim`` READ-ONLY and adds no gate. The
only new perception code is ``perception.realsense_source.RealSenseHandSource``
(one new ``HandSource``). NO hardware, NO ``HW_ENABLE_TOKEN``, NO ``src/comms``.

Public surface:
    teleop_command, drive   — the single per-frame seam + loop (core)
    L20VizModel             — PyBullet GUI/DIRECT renderer (kinematic + mimics)
    run_camera_free         — replay synthetic_openclose through the loop
    run_live                — live RealSense RGB-D mirror
    run_webcam              — live OpenCV/MediaPipe webcam mirror
    run_video               — recorded-video (monocular RGB) mirror
"""
from .core import teleop_command, drive, DEFAULT_DT
from .render import L20VizModel

__all__ = [
    "teleop_command", "drive", "DEFAULT_DT",
    "L20VizModel",
    "run_camera_free", "run_live", "run_webcam", "run_video",
    "replay_stream", "fixture_path", "main",
]

_APP_EXPORTS = {
    "run_camera_free", "run_live", "run_webcam", "run_video",
    "replay_stream", "fixture_path", "main",
}


def __getattr__(name):
    if name in _APP_EXPORTS:
        from . import app
        return getattr(app, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
