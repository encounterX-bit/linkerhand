"""Default real backend: MediaPipe Hands (RGB, monocular).

mediapipe + opencv are heavy and not needed for CI (which runs on synthetic /
recorded streams), so they are imported lazily inside the constructor. Install
them from ``src/perception/requirements.txt`` for live/recorded-video use.

We emit MediaPipe *world* landmarks (metric, origin near the hand centre) so the
frame transform gets a metric cloud; the transform re-origins at the wrist
anyway. z is monocular-estimated, so ``depth_confidence`` is left low -- swap in
an RGB-D source to raise it.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from .indices import FINGER_LANDMARKS, N_LANDMARKS
from .source import HandSource, RawDetection

_TIP_FINGERS = ("thumb", "index", "middle", "ring", "little")


def _parse_fingertip_values(value, *, name: str, signed: bool) -> tuple[float, float, float, float, float]:
    """Return per-finger scalar values ordered thumb,index,middle,ring,little.

    ``value`` may be a scalar or a comma-separated / 5-item sequence ordered as
    thumb,index,middle,ring,little.
    """
    if isinstance(value, str):
        parts = [p.strip() for p in value.split(",")]
        vals = [float(p) for p in parts if p]
    elif isinstance(value, Sequence):
        vals = [float(v) for v in value]
    else:
        vals = [float(value)]
    if len(vals) == 1:
        vals = vals * len(_TIP_FINGERS)
    if len(vals) != len(_TIP_FINGERS):
        raise ValueError(
            f"{name} must be one value or five comma-separated values "
            "ordered thumb,index,middle,ring,little")
    if not signed:
        vals = [max(0.0, v) for v in vals]
    return tuple(vals)


def parse_fingertip_extend(value) -> tuple[float, float, float, float, float]:
    return _parse_fingertip_values(value, name="fingertip_extend", signed=False)


def parse_fingertip_lateral(value) -> tuple[float, float, float, float, float]:
    return _parse_fingertip_values(value, name="fingertip_lateral", signed=True)


def parse_fingertip_straighten(value) -> tuple[float, float, float, float, float]:
    return _parse_fingertip_values(value, name="fingertip_straighten", signed=False)


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else np.zeros_like(v)


def extend_fingertips(landmarks, amount) -> np.ndarray:
    """Move each fingertip farther along its distal bone by ``amount``.

    MediaPipe often places visual tips slightly inside the finger pad. This
    camera-side correction preserves every joint except the five tips and works
    for both image-space (21,2) and world/source-frame (21,3) landmarks.
    """
    return adjust_fingertips(landmarks, extend=amount, lateral=0.0)


def adjust_fingertips(landmarks, *, extend=0.0, lateral=0.0, straighten=0.0) -> np.ndarray:
    """Apply per-finger fingertip length and side-bias corrections.

    ``extend`` moves tips along the distal bone. ``lateral`` moves tips along the
    palm-width direction projected perpendicular to that distal bone; positive is
    roughly toward the index-finger side, negative toward the little-finger side.
    ``straighten`` blends the tip toward the PIP->DIP continuation to counter
    MediaPipe over-bending straight fingers. Values are per-finger ratios.
    """
    extend_amounts = parse_fingertip_extend(extend)
    lateral_amounts = parse_fingertip_lateral(lateral)
    straighten_amounts = parse_fingertip_straighten(straighten)
    pts = np.asarray(landmarks, dtype=float)
    if (
        max(extend_amounts) <= 0.0
        and max(abs(v) for v in lateral_amounts) <= 0.0
        and max(straighten_amounts) <= 0.0
    ):
        return pts.copy()
    if pts.shape[0] != N_LANDMARKS or pts.ndim != 2 or pts.shape[1] not in (2, 3):
        raise ValueError(f"landmarks must be (21,2) or (21,3), got {pts.shape}")

    out = pts.copy()
    side_axis = out[5] - out[17]
    for name, extend_amount, lateral_amount, straighten_amount in zip(
        _TIP_FINGERS,
        extend_amounts,
        lateral_amounts,
        straighten_amounts,
    ):
        _mcp, pip, dip, tip = FINGER_LANDMARKS[name]
        v = out[tip] - out[dip]
        if not np.all(np.isfinite(v)):
            continue
        distal_len = float(np.linalg.norm(v))
        if distal_len <= 1e-9:
            continue
        delta = extend_amount * v
        if abs(lateral_amount) > 0.0:
            u = v / distal_len
            lateral_dir = side_axis - float(side_axis @ u) * u
            lateral_dir = _unit(lateral_dir)
            if not np.any(lateral_dir) and pts.shape[1] == 2:
                lateral_dir = _unit(np.array([-u[1], u[0]], dtype=float))
            delta = delta + lateral_amount * distal_len * lateral_dir
        out[tip] = out[tip] + delta

        if straighten_amount > 0.0:
            pip_to_tip = out[tip] - out[pip]
            pip_to_dip = out[dip] - out[pip]
            if not (np.all(np.isfinite(pip_to_tip)) and np.all(np.isfinite(pip_to_dip))):
                continue
            target_dir = _unit(pip_to_dip)
            target_len = float(np.linalg.norm(pip_to_tip))
            if target_len <= 1e-9 or not np.any(target_dir):
                continue
            target = out[pip] + target_len * target_dir
            blend = min(1.0, max(0.0, straighten_amount))
            out[tip] = (1.0 - blend) * out[tip] + blend * target
    return out


class MediaPipeHandSource(HandSource):
    """Stream hand detections from a video file or live camera via MediaPipe."""

    def __init__(
        self,
        video: Optional[str] = None,
        camera_index: int = 0,
        *,
        max_num_hands: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        depth_confidence: float = 0.3,
        fps: float = 30.0,
        fingertip_extend: float = 0.0,
        fingertip_lateral: float = 0.0,
        fingertip_straighten: float = 0.0,
    ):
        try:
            import cv2  # noqa: F401
            import mediapipe as mp
        except ImportError as e:  # pragma: no cover - depends on optional deps
            raise ImportError(
                "MediaPipeHandSource needs mediapipe + opencv-python. "
                "Install src/perception/requirements.txt. CI uses "
                "SyntheticHandSource / ReplayHandSource instead."
            ) from e

        import cv2

        self._cv2 = cv2
        if video is None and hasattr(cv2, "CAP_V4L2"):
            self._cap = cv2.VideoCapture(camera_index, cv2.CAP_V4L2)
        else:
            self._cap = cv2.VideoCapture(video if video is not None else camera_index)
        if not self._cap.isOpened():
            raise RuntimeError(f"could not open video source {video or camera_index}")
        self._hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.depth_confidence = depth_confidence
        self.fps = fps
        self.fingertip_extend = parse_fingertip_extend(fingertip_extend)
        self.fingertip_lateral = parse_fingertip_lateral(fingertip_lateral)
        self.fingertip_straighten = parse_fingertip_straighten(fingertip_straighten)
        self._frame_idx = 0
        self.last_frame_bgr = None
        self.last_landmarks_px = None
        self.last_landmarks_raw_px = None
        self.last_world_landmarks = None
        self.last_world_landmarks_raw = None

    def read(self) -> RawDetection:
        ok, frame_bgr = self._cap.read()
        if not ok:
            raise StopIteration
        self.last_frame_bgr = frame_bgr.copy()
        t = self._frame_idx / self.fps
        self._frame_idx += 1

        rgb = self._cv2.cvtColor(frame_bgr, self._cv2.COLOR_BGR2RGB)
        res = self._hands.process(rgb)
        world = getattr(res, "multi_hand_world_landmarks", None)
        image = getattr(res, "multi_hand_landmarks", None)
        handed = getattr(res, "multi_handedness", None)
        if not world or not image:
            self.last_landmarks_px = None
            self.last_landmarks_raw_px = None
            self.last_world_landmarks = None
            self.last_world_landmarks_raw = None
            return RawDetection(t=t, ok=False, score=0.0)

        h, w = frame_bgr.shape[:2]
        image_pts = np.array([(lm.x * w, lm.y * h) for lm in image[0].landmark], dtype=float)
        self.last_landmarks_raw_px = [(float(x), float(y)) for x, y in image_pts]
        if (
            max(self.fingertip_extend) > 0.0
            or max(abs(v) for v in self.fingertip_lateral) > 0.0
            or max(self.fingertip_straighten) > 0.0
        ):
            image_pts = adjust_fingertips(
                image_pts,
                extend=self.fingertip_extend,
                lateral=self.fingertip_lateral,
                straighten=self.fingertip_straighten,
            )
        self.last_landmarks_px = [(float(x), float(y)) for x, y in image_pts]
        lms = world[0].landmark
        pts = np.array([[p.x, p.y, p.z] for p in lms], dtype=float)
        if pts.shape != (N_LANDMARKS, 3):
            self.last_landmarks_px = None
            self.last_world_landmarks = None
            self.last_world_landmarks_raw = None
            return RawDetection(t=t, ok=False, score=0.0)
        self.last_world_landmarks_raw = pts.copy()
        if (
            max(self.fingertip_extend) > 0.0
            or max(abs(v) for v in self.fingertip_lateral) > 0.0
            or max(self.fingertip_straighten) > 0.0
        ):
            pts = adjust_fingertips(
                pts,
                extend=self.fingertip_extend,
                lateral=self.fingertip_lateral,
                straighten=self.fingertip_straighten,
            )
        self.last_world_landmarks = pts.copy()

        label, score = None, 0.0
        if handed:
            cls = handed[0].classification[0]
            label, score = cls.label, float(cls.score)
        return RawDetection(
            t=t,
            ok=True,
            landmarks=pts,
            handedness=label,
            score=score,
            depth_confidence=self.depth_confidence,
        )

    def close(self) -> None:  # pragma: no cover - hardware path
        self._cap.release()
        self._hands.close()
