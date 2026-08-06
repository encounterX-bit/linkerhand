"""RGB-D backend: Intel RealSense aligned depth + MediaPipe Hands (metric 3D).

This is the *one* new ``HandSource`` authorised by the viz ticket. It follows the
existing interface exactly (emits ``RawDetection`` in the estimator/source frame);
the rest of perception (``frame.to_hand_base``, the pipeline, one-euro smoothing)
is untouched and consumes it unchanged.

Difference from ``MediaPipeHandSource``: instead of MediaPipe's *estimated*
monocular world-z, every landmark's z is the **measured metric depth** from the
RealSense aligned depth map, deprojected to a 3D point in the colour-camera frame.
That is the whole point of this backend, so ``depth_confidence`` is set **HIGH**
(metric) vs the RGB backend's low monocular value. The palm-plane frame transform
is metric-agnostic and rigid-recovering, so it converts these camera-frame metric
points into the hand_base frame unchanged.

``pyrealsense2`` + ``mediapipe`` + ``opencv-python`` are heavy runtime deps (not in
CI) and are imported lazily inside the constructor — install them from
``src/viz/requirements.txt``. The deprojection / depth-sampling math is factored
into pure module-level helpers (``deproject_pixel``, ``sample_depth_m``,
``deproject_landmarks``) so it is fully testable with mocked intrinsics + a
synthetic depth map, no camera and no heavy deps required.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .indices import N_LANDMARKS
from .source import HandSource, RawDetection

# Depth confidence for a metric RGB-D source. The RGB monocular backend leaves
# this low (~0.3); aligned-depth z is trustworthy, so we flag it HIGH. The
# pipeline's ``depth_warn`` (default 0.5) therefore does NOT raise the
# low-depth-confidence warning for this backend.
HIGH_DEPTH_CONFIDENCE = 1.0

# Last-ditch fill (metres) if a landmark is a depth hole AND there is no
# neighbourhood, no per-landmark history, and no other resolved landmark this
# frame to borrow a depth from. Never emit NaN; a finite-but-flagged point lets
# the pipeline's robustness path decide. ~0.4 m is a typical teleop hand range.
_FALLBACK_DEPTH_M = 0.40


@dataclass
class CameraIntrinsics:
    """Pinhole intrinsics of the colour stream (RealSense ``rs.intrinsics``).

    Aligned-depth-to-colour uses the colour intrinsics; the standard RealSense
    deprojection for the (inverse-)Brown-Conrady model with the aligned colour
    stream reduces to the plain pinhole back-projection implemented here, which
    matches ``rs2_deproject_pixel_to_point`` for that model.
    """

    width: int
    height: int
    fx: float
    fy: float
    ppx: float
    ppy: float

    @classmethod
    def from_rs(cls, intr) -> "CameraIntrinsics":  # pragma: no cover - hw path
        return cls(int(intr.width), int(intr.height),
                   float(intr.fx), float(intr.fy),
                   float(intr.ppx), float(intr.ppy))


def deproject_pixel(intr: CameraIntrinsics, px: float, py: float,
                    depth_m: float) -> np.ndarray:
    """Back-project a pixel + metric depth to a 3D point in the camera frame.

    Camera frame: +x right, +y down, +z forward (out of the lens), metres.
    Equivalent to ``rs2_deproject_pixel_to_point`` for the aligned colour stream.
    """
    x = (px - intr.ppx) / intr.fx
    y = (py - intr.ppy) / intr.fy
    return np.array([depth_m * x, depth_m * y, depth_m], dtype=float)


def sample_depth_m(depth_map: np.ndarray, px: float, py: float,
                   depth_scale: float, *, win: int = 2) -> Tuple[float, bool]:
    """Sample metric depth at a pixel, with small-neighbourhood hole filling.

    ``depth_map`` is the raw aligned depth image (HxW, integer units);
    ``depth_scale`` converts a raw unit to metres (RealSense ``get_depth_scale``).
    Returns ``(depth_m, resolved)``. A zero (or non-finite) raw value is a hole:
    we take the median of non-zero values in a ``(2*win+1)`` window; if the whole
    window is empty, ``resolved`` is False (caller decides the fallback).
    """
    arr = np.asarray(depth_map)
    h, w = arr.shape[:2]
    xi = int(round(px))
    yi = int(round(py))
    xi = 0 if xi < 0 else (w - 1 if xi >= w else xi)
    yi = 0 if yi < 0 else (h - 1 if yi >= h else yi)

    raw = float(arr[yi, xi])
    if np.isfinite(raw) and raw > 0.0:
        return raw * depth_scale, True

    x0, x1 = max(0, xi - win), min(w, xi + win + 1)
    y0, y1 = max(0, yi - win), min(h, yi + win + 1)
    patch = np.asarray(arr[y0:y1, x0:x1], dtype=float)
    nz = patch[np.isfinite(patch) & (patch > 0.0)]
    if nz.size:
        return float(np.median(nz)) * depth_scale, True
    return 0.0, False


def deproject_landmarks(
    landmarks_px: Sequence[Tuple[float, float]],
    depth_map: np.ndarray,
    intr: CameraIntrinsics,
    depth_scale: float,
    *,
    win: int = 2,
    last_depths: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, int, np.ndarray]:
    """Deproject 2D landmark pixels + aligned depth -> metric 3D camera points.

    Depth-hole policy (never emit NaN): (1) sample with neighbourhood median;
    (2) if still a hole, reuse this landmark's last good depth (``last_depths``);
    (3) else borrow the median depth of the resolved landmarks this frame;
    (4) else the constant ``_FALLBACK_DEPTH_M``. Returns
    ``(points (N,3), n_holes, depths (N,))`` — ``depths`` is handed back so the
    caller can carry it forward as the next frame's ``last_depths``.
    """
    n = len(landmarks_px)
    depths = np.empty(n, dtype=float)
    resolved = np.zeros(n, dtype=bool)
    for i, (px, py) in enumerate(landmarks_px):
        depths[i], resolved[i] = sample_depth_m(depth_map, px, py, depth_scale, win=win)

    n_holes = int((~resolved).sum())
    if n_holes:
        frame_fill = float(np.median(depths[resolved])) if resolved.any() else None
        for i in range(n):
            if resolved[i]:
                continue
            if (last_depths is not None and i < len(last_depths)
                    and np.isfinite(last_depths[i]) and last_depths[i] > 0.0):
                depths[i] = float(last_depths[i])
            elif frame_fill is not None:
                depths[i] = frame_fill
            else:
                depths[i] = _FALLBACK_DEPTH_M

    pts = np.array(
        [deproject_pixel(intr, landmarks_px[i][0], landmarks_px[i][1], depths[i])
         for i in range(n)],
        dtype=float,
    )
    return pts, n_holes, depths


class RealSenseHandSource(HandSource):
    """Stream metric hand detections from an Intel RealSense (RGB-D) + MediaPipe.

    Per frame: capture aligned RGB+depth, run MediaPipe Hands on the colour image
    for the 21 2D landmarks, sample the aligned depth at each landmark pixel, and
    deproject to a metric 3D point in the colour-camera frame. Emitted as a
    ``RawDetection`` (source frame); the pipeline transforms it to hand_base.

    If too large a fraction of landmarks are unresolved depth holes
    (``max_hole_frac``), the frame is reported ``ok=False`` so the pipeline holds
    the last good output (reusing the existing robustness path) rather than
    trusting a mostly-filled cloud.
    """

    def __init__(
        self,
        *,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        max_num_hands: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        hole_win: int = 2,
        max_hole_frac: float = 0.5,
        depth_confidence: float = HIGH_DEPTH_CONFIDENCE,
    ):
        try:
            import cv2  # noqa: F401
            import mediapipe as mp
            import pyrealsense2 as rs
        except ImportError as e:  # pragma: no cover - depends on optional deps
            raise ImportError(
                "RealSenseHandSource needs pyrealsense2 + mediapipe + "
                "opencv-python. Install src/viz/requirements.txt. CI/tests use "
                "the pure deproject helpers + SyntheticHandSource instead."
            ) from e

        self._cv2 = cv2
        self._rs = rs
        self.depth_confidence = depth_confidence
        self.hole_win = hole_win
        self.max_hole_frac = max_hole_frac
        self.fps = fps
        self._frame_idx = 0
        self._last_depths: Optional[np.ndarray] = None
        self.last_frame_bgr = None
        self.last_landmarks_px = None

        # RealSense pipeline: depth + colour, aligned to colour.
        self._pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        profile = self._pipeline.start(cfg)
        self._align = rs.align(rs.stream.color)
        self._depth_scale = float(
            profile.get_device().first_depth_sensor().get_depth_scale())

        self._hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    # -- per-frame assembly (pure given inputs; covered by tests) ---------- #
    def _build_detection(
        self,
        landmarks_px: Sequence[Tuple[float, float]],
        depth_map: np.ndarray,
        intr: CameraIntrinsics,
        t: float,
        handed_label: Optional[str],
        score: float,
    ) -> RawDetection:
        pts, n_holes, depths = deproject_landmarks(
            landmarks_px, depth_map, intr, self._depth_scale,
            win=self.hole_win, last_depths=self._last_depths,
        )
        self._last_depths = depths
        # Too many holes -> don't trust the cloud; let the pipeline hold last good.
        if n_holes > self.max_hole_frac * len(landmarks_px):
            return RawDetection(t=t, ok=False, score=0.0)
        return RawDetection(
            t=t,
            ok=True,
            landmarks=pts,
            handedness=handed_label,
            score=score,
            depth_confidence=self.depth_confidence,
        )

    def read(self) -> RawDetection:  # pragma: no cover - hardware path
        frames = self._pipeline.wait_for_frames()
        frames = self._align.process(frames)
        color = frames.get_color_frame()
        depth = frames.get_depth_frame()
        t = self._frame_idx / float(self.fps)
        self._frame_idx += 1
        if not color or not depth:
            return RawDetection(t=t, ok=False, score=0.0)

        intr = CameraIntrinsics.from_rs(
            color.get_profile().as_video_stream_profile().get_intrinsics())
        color_img = np.asanyarray(color.get_data())
        self.last_frame_bgr = color_img.copy()
        depth_map = np.asanyarray(depth.get_data())

        rgb = self._cv2.cvtColor(color_img, self._cv2.COLOR_BGR2RGB)
        res = self._hands.process(rgb)
        lms = getattr(res, "multi_hand_landmarks", None)   # image-space (normalised)
        handed = getattr(res, "multi_handedness", None)
        if not lms:
            self.last_landmarks_px = None
            return RawDetection(t=t, ok=False, score=0.0)

        h, w = depth_map.shape[:2]
        pts_px = [(lm.x * w, lm.y * h) for lm in lms[0].landmark]
        if len(pts_px) != N_LANDMARKS:
            self.last_landmarks_px = None
            return RawDetection(t=t, ok=False, score=0.0)
        self.last_landmarks_px = pts_px

        label, score = None, 0.0
        if handed:
            cls = handed[0].classification[0]
            label, score = cls.label, float(cls.score)

        return self._build_detection(pts_px, depth_map, intr, t, label, score)

    def close(self) -> None:  # pragma: no cover - hardware path
        try:
            self._pipeline.stop()
        finally:
            self._hands.close()
