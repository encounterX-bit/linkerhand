"""``HandSource`` interface + estimator-agnostic backends.

The pipeline consumes ``RawDetection`` objects from any ``HandSource``. MediaPipe
Hands (RGB, monocular) is the default real backend (``mediapipe_source.py``); an
RGB-D source or a MANO estimator (HaMeR/WiLoR) can drop in by emitting the same
``RawDetection`` -- the output contract never changes. The synthetic/replay
sources here keep CI camera-free.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Iterator, List, Optional, Sequence

import numpy as np

from .indices import CANONICAL_FLAT_RIGHT


@dataclass
class RawDetection:
    """One frame from an estimator, in the estimator's own (source) frame.

    ``ok=False`` means no hand was detected this frame (``landmarks`` may be
    None); the pipeline holds the last good output and flags it. ``landmarks``
    are metric 3D when available (MediaPipe *world* landmarks); ``score`` is the
    detection/tracking confidence in [0, 1]. ``depth_confidence`` flags how
    trustworthy z is -- low for monocular, ~1.0 for aligned RGB-D.
    """

    t: float
    ok: bool = True
    landmarks: Optional[np.ndarray] = None      # (21, 3) source frame
    handedness: Optional[str] = None            # raw estimator label
    score: float = 0.0
    depth_confidence: float = 0.3


class HandSource(ABC):
    """A stream of per-frame hand detections.

    ``read()`` returns the next ``RawDetection`` (possibly ``ok=False``) or
    raises ``StopIteration`` when the stream ends. Sources are iterable.
    """

    @abstractmethod
    def read(self) -> RawDetection:
        ...

    def close(self) -> None:  # pragma: no cover - trivial default
        pass

    def __iter__(self) -> Iterator[RawDetection]:
        return self

    def __next__(self) -> RawDetection:
        return self.read()


class ReplayHandSource(HandSource):
    """Replay a fixed list of ``RawDetection`` (e.g. a recorded session)."""

    def __init__(self, detections: Sequence[RawDetection]):
        self._dets = list(detections)
        self._i = 0

    def read(self) -> RawDetection:
        if self._i >= len(self._dets):
            raise StopIteration
        d = self._dets[self._i]
        self._i += 1
        return d


def _curl_finger(base4: np.ndarray, mcp: np.ndarray, amount: float) -> np.ndarray:
    """Curl a 4-point finger [mcp,pip,dip,tip] toward -x by ``amount`` radians,
    flexing about the +y (radial) axis -- matches the hand_base convention."""
    pts = [mcp.copy()]
    prev = mcp.copy()
    cum = 0.0
    # progressive flexion: each joint adds curl, like a real closing fist.
    for k in range(1, 4):
        seg = base4[k] - base4[k - 1]
        cum += amount
        c, s = np.cos(cum), np.sin(cum)
        # rotation about +y: x' = c*x + s*z ; z' = -s*x + c*z  (curls toward -x)
        rx = c * seg[0] + s * seg[2]
        rz = -s * seg[0] + c * seg[2]
        seg = np.array([rx, seg[1], rz])
        prev = prev + seg
        pts.append(prev.copy())
    return np.array(pts)


class SyntheticHandSource(HandSource):
    """A camera-free moving hand for tests and the synthetic 'real' fixture.

    Animates an open<->close motion from a canonical hand, places it in an
    arbitrary *camera* frame (rotation/translation/scale), adds Gaussian noise
    (heavier on z, like monocular depth), and optionally drops random frames
    (``ok=False``) and dips confidence -- exercising the full pipeline.
    """

    def __init__(
        self,
        n_frames: int = 60,
        side: str = "right",
        *,
        fps: float = 30.0,
        noise: float = 0.0,
        z_noise_mult: float = 3.0,
        camera_R: Optional[np.ndarray] = None,
        camera_t: Optional[np.ndarray] = None,
        camera_scale: float = 1.0,
        dropout_frames: Sequence[int] = (),
        low_conf_frames: Sequence[int] = (),
        seed: int = 0,
        depth_confidence: float = 0.3,
        base_landmarks: Optional[np.ndarray] = None,
    ):
        self.n_frames = n_frames
        self.side = side
        self.fps = fps
        self.noise = noise
        self.z_noise_mult = z_noise_mult
        self.camera_R = np.eye(3) if camera_R is None else np.asarray(camera_R, float)
        self.camera_t = np.zeros(3) if camera_t is None else np.asarray(camera_t, float)
        self.camera_scale = camera_scale
        self.dropout = set(dropout_frames)
        self.low_conf = set(low_conf_frames)
        self.depth_confidence = depth_confidence
        self._rng = np.random.default_rng(seed)
        self._i = 0
        base = CANONICAL_FLAT_RIGHT if base_landmarks is None else np.asarray(base_landmarks, float)
        self._base = base.copy()
        if side == "left":
            self._base = self._base.copy()
            self._base[:, 1] *= -1.0
        # MediaPipe reports camera-view labels; with a non-mirrored image the
        # physical 'right' hand is labelled 'Left'. We emit the *swapped* label
        # so the pipeline (image_mirrored=False) recovers the physical side.
        self._raw_label = "Left" if side == "right" else "Right"

    def _pose(self, frac: float) -> np.ndarray:
        """Open (frac=0) -> fist (frac=1) in the hand_base frame."""
        from .indices import FINGER_LANDMARKS, MIDDLE_MCP

        lm = self._base.copy()
        amount = 1.2 * frac
        for name in ("index", "middle", "ring", "little"):
            a, b, c, d = FINGER_LANDMARKS[name]
            base4 = self._base[[a, b, c, d]]
            curled = _curl_finger(base4, self._base[a], amount)
            lm[[a, b, c, d]] = curled
        return lm

    def read(self) -> RawDetection:
        if self._i >= self.n_frames:
            raise StopIteration
        i = self._i
        self._i += 1
        t = i / self.fps

        if i in self.dropout:
            return RawDetection(t=t, ok=False, score=0.0)

        # triangular open/close motion
        frac = 1.0 - abs(2.0 * (i / max(1, self.n_frames - 1)) - 1.0)
        lm = self._pose(frac)

        # into the camera frame: scale, rotate, translate
        lm_cam = (self.camera_scale * lm) @ self.camera_R.T + self.camera_t
        if self.noise > 0.0:
            n = self._rng.normal(0.0, self.noise, size=lm_cam.shape)
            n[:, 2] *= self.z_noise_mult
            lm_cam = lm_cam + n

        score = 0.3 if i in self.low_conf else 0.95
        return RawDetection(
            t=t,
            ok=True,
            landmarks=lm_cam,
            handedness=self._raw_label,
            score=score,
            depth_confidence=self.depth_confidence,
        )
