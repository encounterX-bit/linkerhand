"""Video-file backend: a recorded hand clip -> MediaPipe -> ``RawDetection``.

This is an additive ``HandSource`` for the viz loop's third input source. It
**reuses ``MediaPipeHandSource``'s RGB detection path verbatim** (subclass) — the
ONLY difference from the webcam backend is frame acquisition + timing: frames come
from a video file via ``cv2.VideoCapture(path)`` and the stream honours the clip's
**native FPS** (with an optional playback-rate / fps override) instead of assuming
a fixed camera rate.

It is monocular RGB (no depth), so ``depth_confidence`` stays **LOW**, identical to
``MediaPipeHandSource``. That means this validates the end-to-end *plumbing* — real
(non-synthetic) hand motion -> MediaPipe -> palm-plane frame -> ``retarget()`` ->
``safety.filter()`` -> sim joints — NOT depth/retarget accuracy (that is the
RealSense path). No-detection / low-confidence frames hold the last good output in
the pipeline (never NaN); end-of-file raises ``StopIteration`` so the loop stops
cleanly.

``mediapipe`` + ``opencv-python`` are heavy and not in CI; they are imported lazily
by the parent constructor (see ``src/perception/requirements.txt``). The FPS
resolution is a pure module-level helper (:func:`resolve_fps`) and the per-frame
parsing is the inherited ``read()``, so the read loop is testable with a fake
capture + fake detector and no heavy deps.
"""
from __future__ import annotations

import math
import os
from typing import Optional

from .mediapipe_source import MediaPipeHandSource

# Monocular RGB z is hand-relative and noisy -> the same LOW confidence the webcam
# backend reports. The pipeline's ``depth_warn`` (default 0.5) therefore raises the
# low-depth-confidence warning, exactly as for ``MediaPipeHandSource``.
LOW_DEPTH_CONFIDENCE = 0.3

# Fallback frame rate when the container reports no usable FPS metadata.
_DEFAULT_FPS = 30.0


def resolve_fps(native_fps: Optional[float], override: Optional[float] = None,
                default: float = _DEFAULT_FPS) -> float:
    """Pick the timestamp FPS: explicit override > container-native > default.

    Many containers report 0 (or a NaN/inf) when FPS metadata is missing; such
    values are rejected in favour of ``default`` so per-frame timestamps
    (``t = frame_idx / fps``, the one-euro smoother's dt) are always sane.
    """
    if override is not None and math.isfinite(override) and override > 0.0:
        return float(override)
    if native_fps is not None and math.isfinite(native_fps) and native_fps > 0.0:
        return float(native_fps)
    return float(default)


class VideoHandSource(MediaPipeHandSource):
    """Stream hand detections from a recorded video file via MediaPipe.

    Differs from :class:`MediaPipeHandSource` only in frame acquisition + timing:
    it opens a video file (not a webcam) and derives the timestamp FPS from the
    clip's native rate (overridable). Detection, landmark extraction and the
    ``RawDetection`` contract are the inherited, unmodified MediaPipe path.
    """

    def __init__(
        self,
        path: str,
        *,
        fps: Optional[float] = None,
        playback_rate: float = 1.0,
        max_num_hands: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        depth_confidence: float = LOW_DEPTH_CONFIDENCE,
        fingertip_extend: float = 0.0,
        fingertip_lateral: float = 0.0,
        fingertip_straighten: float = 0.0,
    ):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"video file not found: {path}")

        # Parent opens cv2.VideoCapture(path) + MediaPipe Hands (lazy heavy import).
        super().__init__(
            video=path,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            depth_confidence=depth_confidence,
            fps=_DEFAULT_FPS,  # overwritten below from the container's native FPS
            fingertip_extend=fingertip_extend,
            fingertip_lateral=fingertip_lateral,
            fingertip_straighten=fingertip_straighten,
        )
        self.path = path
        self.playback_rate = float(playback_rate)

        native = float(self._cap.get(self._cv2.CAP_PROP_FPS))
        self.native_fps = resolve_fps(native)
        # Timestamps drive the one-euro smoother's dt -> use the native (or
        # overridden) capture rate, NOT the wall-clock playback rate.
        self.fps = resolve_fps(native, override=fps)
        try:
            self.frame_count = int(self._cap.get(self._cv2.CAP_PROP_FRAME_COUNT))
        except Exception:  # pragma: no cover - container without a frame count
            self.frame_count = -1

    @property
    def frame_period(self) -> float:
        """Wall-clock seconds the viz loop should hold each frame for playback.

        ``playback_rate`` > 1 plays faster (shorter hold), < 1 slower. This only
        affects display pacing in the loop; the emitted timestamps already use the
        native FPS so retargeting/smoothing see true motion timing.
        """
        rate = self.playback_rate if self.playback_rate > 0.0 else 1.0
        return 1.0 / (self.fps * rate)
