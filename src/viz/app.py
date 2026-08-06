"""Run a live camera -> sim mirror, or the camera-free replay, in one loop.

``run_camera_free`` replays the committed ``synthetic_openclose`` sequence and
``run_live`` reads a RealSense camera; all sources feed the *same* ``core.drive``
loop (retarget -> filter -> kinematic joint set), so the camera-free path is a
faithful dry-run of the live mirror and is what the headless equivalence test
drives.

Stage 1 only: kinematic mirror, no dynamics, no contact, no hardware.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from typing import Iterator, List, Optional, Tuple

import numpy as np

from .core import DEFAULT_DT, drive
from .render import L20VizModel

_THIS = os.path.dirname(os.path.abspath(__file__))
_REAL_FIXTURES = os.path.join(
    os.path.dirname(_THIS), os.pardir, "tests", "g1_kinematic", "fixtures", "real")


_HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
)


def _one_euro_config(min_cutoff: float, beta: float, d_cutoff: float):
    from src.perception.one_euro import OneEuroConfig

    return OneEuroConfig(
        min_cutoff=float(min_cutoff),
        beta=float(beta),
        d_cutoff=float(d_cutoff),
    )


def _draw_hand_overlay(cv2, view, points_px) -> None:
    if points_px is None or len(points_px) < 21:
        return
    h, w = view.shape[:2]
    pts = []
    for px, py in points_px[:21]:
        if not np.isfinite(px) or not np.isfinite(py):
            return
        x = max(0, min(w - 1, int(round(px))))
        y = max(0, min(h - 1, int(round(py))))
        pts.append((x, y))
    for a, b in _HAND_CONNECTIONS:
        cv2.line(view, pts[a], pts[b], (0, 255, 255), 2, cv2.LINE_AA)
    for i, p in enumerate(pts):
        color = (0, 180, 255) if i in (4, 8, 12, 16, 20) else (0, 255, 0)
        cv2.circle(view, p, 4, color, -1, cv2.LINE_AA)


def _camera_preview(cv2, source, window: str, pf) -> bool:
    """Show the source's latest BGR frame. Return False when the user quits."""
    if cv2 is None:
        return True
    frame = getattr(source, "last_frame_bgr", None)
    if frame is None:
        return True
    view = frame.copy()
    _draw_hand_overlay(cv2, view, getattr(source, "last_landmarks_px", None))
    if pf is None:
        text = "no hand"
    else:
        status = "held" if getattr(pf, "held", False) else "detected"
        text = f"{pf.side}  {status}  score={pf.score:.2f}"
    cv2.putText(view, text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow(window, view)
    key = cv2.waitKey(1) & 0xFF
    return key not in (27, ord("q"))


def _thumb_adjuster(gain: float, cross_gain: float, assist_smooth: float = 0.0):
    """Return a sim-only thumb adjuster.

    ``gain`` is the explicit legacy multiplier. ``cross_gain`` pushes the thumb
    opposition farther across the palm when the human thumb tip approaches the
    four-finger MCP root line.
    """
    gain = float(gain)
    cross_gain = float(cross_gain)
    assist_smooth = max(0.0, min(0.95, float(assist_smooth)))
    if abs(gain - 1.0) < 1e-12 and abs(cross_gain) < 1e-12:
        return None

    from src.finger_retarget.constants import CONSTANTS
    state = {"blend": None, "t": None}

    def _adjust(candidate: dict, *, landmarks=None) -> dict:
        side = candidate.get("side", "right")
        thumb = CONSTANTS[side]["thumb"]
        thumb_limits = {int(idx): tuple(lim) for idx, _axis, lim in thumb["base_axes"]}
        thumb_limits[int(thumb["tip_idx"])] = tuple(thumb["tip_limit"])
        base_idx = int(thumb["base_idx"])
        abd_idx = int(thumb["abd_idx"])
        opp_idx = int(thumb["opp_idx"])
        tip_idx = int(thumb["tip_idx"])

        q = list(candidate["joint_rad"])
        for idx, (lo, hi) in thumb_limits.items():
            q[idx] = max(lo, min(hi, q[idx] * gain))

        if cross_gain > 0.0 and landmarks is not None:
            lm = np.asarray(landmarks, dtype=float)
            if lm.shape == (21, 3) and np.all(np.isfinite(lm)):
                roots = lm[[5, 9, 13, 17]]
                tip = lm[4]
                root_line = roots[-1] - roots[0]
                line_len2 = float(root_line @ root_line)
                if line_len2 > 1e-10:
                    t = float(((tip - roots[0]) @ root_line) / line_len2)
                    t = max(0.0, min(1.0, t))
                    closest = roots[0] + t * root_line
                    palm_width = float(np.sqrt(line_len2))
                    line_dist = float(np.linalg.norm(tip - closest))
                    contact = (0.75 * palm_width - line_dist) / (0.45 * palm_width)
                    contact = max(0.0, min(1.0, contact))
                    blend = max(0.0, min(1.0, cross_gain * contact))
                    if assist_smooth > 0.0:
                        prev_blend = state["blend"]
                        prev_t = state["t"]
                        if prev_blend is not None:
                            blend = assist_smooth * float(prev_blend) + (1.0 - assist_smooth) * blend
                        if prev_t is not None:
                            t = assist_smooth * float(prev_t) + (1.0 - assist_smooth) * t
                        state["blend"] = blend
                        state["t"] = t

                    targets = {
                        base_idx: 0.60 + 0.19 * t,
                        abd_idx: 0.90 + 0.30 * t,
                        opp_idx: 0.34 + 0.98 * t,
                        tip_idx: 0.78 + 0.26 * t,
                    }
                    for idx, target in targets.items():
                        lo, hi = thumb_limits[idx]
                        target = max(lo, min(hi, target))
                        q[idx] = max(lo, min(hi, (1.0 - blend) * q[idx] + blend * target))

        out = dict(candidate)
        out["joint_rad"] = q
        return out

    return _adjust


def _little_abd_adjuster(gain: float):
    """Optionally scale only little-finger side orientation (q9)."""
    gain = float(gain)
    if abs(gain - 1.0) < 1e-12:
        return None

    from src.finger_retarget.constants import CONSTANTS

    def _adjust(candidate: dict, *, landmarks=None) -> dict:
        side = candidate.get("side", "right")
        little = CONSTANTS[side]["little"]
        idx = int(little["abd_idx"])
        lo, hi = tuple(lim for joint_idx, _axis, lim in little["base_axes"]
                       if int(joint_idx) == idx)[0]
        q = list(candidate["joint_rad"])
        q[idx] = max(lo, min(hi, q[idx] * gain))
        out = dict(candidate)
        out["joint_rad"] = q
        return out

    return _adjust


def _thumb_grasp_adjuster(gain: float):
    """Add a thumb-only grasp response for sim object grasping.

    This does not move the four fingers. It leaves the raw camera retarget as the
    seed and adds opposition/tip curl only when the human thumb is actually inside
    the palm/fingertip interaction region.
    """
    gain = float(gain)
    if gain <= 0.0:
        return None

    from src.finger_retarget.constants import CONSTANTS

    def _adjust(candidate: dict, *, landmarks=None) -> dict:
        if landmarks is None:
            return candidate
        lm = np.asarray(landmarks, dtype=float)
        if lm.shape != (21, 3) or not np.all(np.isfinite(lm)):
            return candidate

        roots = lm[[5, 9, 13, 17]]
        root_line = roots[-1] - roots[0]
        line_len2 = float(root_line @ root_line)
        palm_width = float(np.sqrt(line_len2))
        if palm_width < 1e-9:
            return candidate

        thumb_tip = lm[4]
        t = float(((thumb_tip - roots[0]) @ root_line) / line_len2)
        t = max(0.0, min(1.0, t))
        closest = roots[0] + t * root_line
        root_line_contact = (0.80 * palm_width - float(np.linalg.norm(thumb_tip - closest))) / (0.55 * palm_width)
        root_line_contact = max(0.0, min(1.0, root_line_contact))

        nearest_tip = min(float(np.linalg.norm(thumb_tip - lm[i])) for i in (8, 12, 16, 20))
        tip_contact = (0.90 * palm_width - nearest_tip) / (0.55 * palm_width)
        tip_contact = max(0.0, min(1.0, tip_contact))
        activation = max(root_line_contact, tip_contact)
        if activation <= 0.0:
            return candidate
        activation = activation ** 0.7

        v23 = lm[3] - lm[2]
        v34 = lm[4] - lm[3]
        n23 = float(np.linalg.norm(v23))
        n34 = float(np.linalg.norm(v34))
        distal_bend = 0.0
        if n23 > 1e-9 and n34 > 1e-9:
            cos_tip = float((v23 @ v34) / (n23 * n34))
            cos_tip = max(-1.0, min(1.0, cos_tip))
            distal_bend = max(0.0, min(1.0, float(np.arccos(cos_tip)) / 0.9))

        side = candidate.get("side", "right")
        thumb = CONSTANTS[side]["thumb"]
        limits = {int(idx): tuple(lim) for idx, _axis, lim in thumb["base_axes"]}
        limits[int(thumb["tip_idx"])] = tuple(thumb["tip_limit"])
        base_idx = int(thumb["base_idx"])
        abd_idx = int(thumb["abd_idx"])
        opp_idx = int(thumb["opp_idx"])
        tip_idx = int(thumb["tip_idx"])

        q = list(candidate["joint_rad"])
        deltas = {
            base_idx: 0.10 * activation,
            abd_idx: 0.20 * activation,
            opp_idx: (0.42 + 0.20 * t) * activation,
            tip_idx: 0.08 * activation + 0.14 * distal_bend,
        }
        scale = max(0.0, min(1.0, gain))
        for idx, delta in deltas.items():
            lo, hi = limits[idx]
            q[idx] = max(lo, min(hi, q[idx] + scale * delta))

        out = dict(candidate)
        out["joint_rad"] = q
        return out

    return _adjust


def _thumb_base_assist_adjuster(gain: float, smooth: float = 0.0):
    """Add front-fingertip thumb base reach without increasing side sweep."""
    gain = float(gain)
    smooth = max(0.0, min(0.95, float(smooth)))
    if gain <= 0.0:
        return None

    from src.finger_retarget.constants import CONSTANTS

    state = {"activation": None}

    def _adjust(candidate: dict, *, landmarks=None) -> dict:
        if landmarks is None:
            return candidate
        lm = np.asarray(landmarks, dtype=float)
        if lm.shape != (21, 3) or not np.all(np.isfinite(lm)):
            return candidate

        roots = lm[[5, 9, 13, 17]]
        root_line = roots[-1] - roots[0]
        palm_width = float(np.linalg.norm(root_line))
        if palm_width < 1e-9:
            return candidate

        thumb_tip = lm[4]
        front_dist = min(float(np.linalg.norm(thumb_tip - lm[i])) for i in (8, 12))
        back_dist = min(float(np.linalg.norm(thumb_tip - lm[i])) for i in (16, 20))
        front_contact = (0.82 * palm_width - front_dist) / (0.52 * palm_width)
        front_contact = max(0.0, min(1.0, front_contact))
        front_bias = (back_dist - front_dist + 0.18 * palm_width) / (0.34 * palm_width)
        front_bias = max(0.0, min(1.0, front_bias))
        activation = (front_contact * front_bias) ** 0.7
        if smooth > 0.0:
            prev = state["activation"]
            if prev is not None:
                activation = smooth * float(prev) + (1.0 - smooth) * activation
            state["activation"] = activation
        if activation <= 1e-6:
            return candidate

        side = candidate.get("side", "right")
        thumb = CONSTANTS[side]["thumb"]
        limits = {int(idx): tuple(lim) for idx, _axis, lim in thumb["base_axes"]}
        base_idx = int(thumb["base_idx"])

        q = list(candidate["joint_rad"])
        lo, hi = limits[base_idx]
        q[base_idx] = max(lo, min(hi, q[base_idx] + 0.26 * gain * activation))

        out = dict(candidate)
        out["joint_rad"] = q
        return out

    return _adjust


def _thumb_tip_adjuster(gain: float):
    """Scale only the thumb tip curl q15 after the other thumb assists."""
    gain = float(gain)
    if abs(gain - 1.0) < 1e-12:
        return None

    from src.finger_retarget.constants import CONSTANTS

    def _adjust(candidate: dict, *, landmarks=None) -> dict:
        side = candidate.get("side", "right")
        thumb = CONSTANTS[side]["thumb"]
        tip_idx = int(thumb["tip_idx"])
        lo, hi = tuple(thumb["tip_limit"])
        q = list(candidate["joint_rad"])
        q[tip_idx] = max(lo, min(hi, q[tip_idx] * gain))
        out = dict(candidate)
        out["joint_rad"] = q
        return out

    return _adjust


def _thumb_orient_adjuster(gain: float):
    """Use distal thumb direction only near grasp/contact regions.

    Open thumb spread should mostly stay raw retarget output. This helper is
    opt-in and only adds a small thumb-only correction once the human thumb tip
    is actually near the four-finger MCP line or fingertips.
    """
    gain = float(gain)
    if gain <= 0.0:
        return None

    from src.finger_retarget.constants import CONSTANTS

    def _adjust(candidate: dict, *, landmarks=None) -> dict:
        if landmarks is None:
            return candidate
        lm = np.asarray(landmarks, dtype=float)
        if lm.shape != (21, 3) or not np.all(np.isfinite(lm)):
            return candidate

        roots = lm[[5, 9, 13, 17]]
        root_line = roots[-1] - roots[0]
        palm_width = float(np.linalg.norm(root_line))
        if palm_width < 1e-9:
            return candidate
        v23 = lm[3] - lm[2]
        v34 = lm[4] - lm[3]
        n23 = float(np.linalg.norm(v23))
        n34 = float(np.linalg.norm(v34))
        distal_bend = 0.0
        if n23 > 1e-9 and n34 > 1e-9:
            cos_tip = float((v23 @ v34) / (n23 * n34))
            cos_tip = max(-1.0, min(1.0, cos_tip))
            distal_bend = max(0.0, min(1.0, float(np.arccos(cos_tip)) / 0.9))

        thumb_tip = lm[4]
        t = float(((lm[4] - roots[0]) @ root_line) / (palm_width * palm_width))
        reach = max(0.0, min(1.0, t))

        closest = roots[0] + reach * root_line
        root_contact = (0.72 * palm_width - float(np.linalg.norm(thumb_tip - closest))) / (0.42 * palm_width)
        root_contact = max(0.0, min(1.0, root_contact))

        nearest_tip = min(float(np.linalg.norm(thumb_tip - lm[i])) for i in (8, 12, 16, 20))
        tip_contact = (0.72 * palm_width - nearest_tip) / (0.42 * palm_width)
        tip_contact = max(0.0, min(1.0, tip_contact))

        drive = max(root_contact, tip_contact) ** 0.8
        if drive <= 0.0:
            return candidate

        side = candidate.get("side", "right")
        thumb = CONSTANTS[side]["thumb"]
        limits = {int(idx): tuple(lim) for idx, _axis, lim in thumb["base_axes"]}
        limits[int(thumb["tip_idx"])] = tuple(thumb["tip_limit"])
        abd_idx = int(thumb["abd_idx"])
        tip_idx = int(thumb["tip_idx"])

        q = list(candidate["joint_rad"])

        nonthumb_base = [1, 2, 3, 4]
        nonthumb_closed = max(0.0, min(1.0, sum(q[i] for i in nonthumb_base) / (len(nonthumb_base) * 0.95)))
        drive *= 1.0 - 0.85 * nonthumb_closed
        if drive <= 1e-6:
            return candidate

        scale = max(0.0, min(1.0, gain))
        raw_t2 = max(0.0, min(1.0, float(((lm[2] - roots[0]) @ root_line) / (palm_width * palm_width))))
        raw_t3 = max(0.0, min(1.0, float(((lm[3] - roots[0]) @ root_line) / (palm_width * palm_width))))
        raw_t4 = reach
        front_sweep = max(0.0, min(1.0, 0.10 * raw_t2 + 0.25 * raw_t3 + 0.65 * raw_t4))
        front_dir = max(0.0, min(1.0, (raw_t4 - raw_t2) / 0.60))
        side_swing = max(0.0, min(1.0, 0.35 * front_sweep + 0.65 * front_dir)) ** 0.55

        # Keep this as an orientation polish. Grasp/opposition is owned by the
        # retargeter and _thumb_grasp_adjuster; pushing q0/q10 here made the
        # thumb snap sideways and fight the tip response near contact.
        deltas = {
            abd_idx: 0.42 * side_swing * drive,
            tip_idx: 0.10 * distal_bend * drive,
        }
        for idx, delta in deltas.items():
            lo, hi = limits[idx]
            q[idx] = max(lo, min(hi, q[idx] + scale * delta))

        out = dict(candidate)
        out["joint_rad"] = q
        return out

    return _adjust


def _thumb_only_adjuster(enabled: bool):
    """Freeze all non-thumb joints to open so thumb mapping can be tuned alone."""
    if not enabled:
        return None

    nonthumb_idx = (1, 2, 3, 4, 6, 7, 8, 9, 16, 17, 18, 19)

    def _adjust(candidate: dict, *, landmarks=None) -> dict:
        q = list(candidate["joint_rad"])
        for idx in nonthumb_idx:
            q[idx] = 0.0
        out = dict(candidate)
        out["joint_rad"] = q
        return out

    return _adjust


def _compose_adjust(*adjusters):
    active = [a for a in adjusters if a is not None]
    if not active:
        return None
    if len(active) == 1:
        return active[0]

    def _adjust(candidate: dict, *, landmarks=None) -> dict:
        for adjuster in active:
            candidate = adjuster(candidate, landmarks=landmarks)
        return candidate

    return _adjust


def _match_debug(enabled: bool, period: int):
    if not enabled:
        return None
    from src.viz.match_debug import make_match_debugger
    return make_match_debugger(True, period=period)


def fixture_path(side: str) -> str:
    return os.path.normpath(
        os.path.join(_REAL_FIXTURES, f"synthetic_openclose_{side}.json"))


def replay_stream(side: str = "right") -> Iterator[Tuple[str, np.ndarray, float]]:
    """Yield ``(side, landmarks(21,3), t)`` from a committed synthetic_openclose
    fixture (already hand_base frame — the pipeline output), exactly the input the
    G1/G2 pipeline consumes. No perception re-run, so the replay matches the
    existing pipeline on identical landmarks."""
    with open(fixture_path(side)) as f:
        data = json.load(f)
    fixture_side = data.get("side", side)
    for fr in data["frames"]:
        yield fixture_side, np.asarray(fr["landmarks"], dtype=float), float(fr["t"])


# --------------------------------------------------------------------------- #
# Camera-free replay (default / testable)
# --------------------------------------------------------------------------- #
def run_camera_free(side: str = "right", *, gui: bool = True, use_filter: bool = True,
                    loop: bool = False, realtime: Optional[bool] = None,
                    dt: float = DEFAULT_DT, max_frames: Optional[int] = None,
                    thumb_gain: float = 1.0, thumb_cross_gain: float = 1.0,
                    thumb_assist_smooth: float = 0.0,
                    little_abd_gain: float = 1.0,
                    thumb_grasp_gain: float = 0.0,
                    thumb_base_assist_gain: float = 0.0,
                    thumb_tip_gain: float = 1.0,
                    thumb_orient_gain: float = 0.0,
                    thumb_only: bool = False,
                    debug_match: bool = False,
                    debug_match_period: int = 15) -> List[dict]:
    """Replay synthetic_openclose through the identical viz loop.

    ``gui=False`` runs headless (DIRECT) for tests. ``realtime`` defaults to the
    GUI setting (sleep to ~camera rate only when showing a window). Returns the
    list of per-frame records from :func:`core.drive`.
    """
    if realtime is None:
        realtime = gui
    model = L20VizModel(side, gui=gui)
    adjust = _compose_adjust(
        _thumb_adjuster(thumb_gain, thumb_cross_gain, thumb_assist_smooth),
        _little_abd_adjuster(little_abd_gain),
        _thumb_grasp_adjuster(thumb_grasp_gain),
        _thumb_base_assist_adjuster(thumb_base_assist_gain, thumb_assist_smooth),
        _thumb_orient_adjuster(thumb_orient_gain),
        _thumb_tip_adjuster(thumb_tip_gain),
        _thumb_only_adjuster(thumb_only),
    )
    records: List[dict] = []
    try:
        while True:
            stream = replay_stream(side)
            if max_frames is not None:
                stream = (x for i, x in enumerate(stream) if i < max_frames)
            if realtime:
                def _sleeper(it):
                    for x in it:
                        yield x
                        time.sleep(dt)
                stream = _sleeper(stream)
            records = drive(model, stream, use_filter=use_filter, dt=dt,
                            candidate_adjust=adjust,
                            debug_callback=_match_debug(debug_match, debug_match_period))
            if not loop:
                break
    finally:
        model.close()
    return records


# --------------------------------------------------------------------------- #
# Live RealSense mirror
# --------------------------------------------------------------------------- #
def run_live(side: Optional[str] = None, *, gui: bool = True, use_filter: bool = True,
             dt: float = DEFAULT_DT, image_mirrored: bool = False,
             show_camera: bool = False, max_frames: Optional[int] = None,
             smoothing: bool = True,
             one_euro_min_cutoff: float = 1.5,
             one_euro_beta: float = 0.05,
             one_euro_d_cutoff: float = 1.0,
             thumb_gain: float = 1.0, thumb_cross_gain: float = 1.0,
             thumb_assist_smooth: float = 0.0,
             little_abd_gain: float = 1.0,
             thumb_grasp_gain: float = 0.0,
             thumb_base_assist_gain: float = 0.0,
             thumb_tip_gain: float = 1.0,
             thumb_orient_gain: float = 0.0,
             thumb_only: bool = False,
             debug_match: bool = False,
             debug_match_period: int = 15) -> List[dict]:
    """Live mirror: RealSense RGB-D -> perception pipeline -> viz loop.

    Builds a ``RealSenseHandSource`` (metric depth) and the existing
    ``HandPipeline`` (palm-plane transform + one-euro smoothing on), then drives
    the same loop. ``side`` is normally resolved per-frame from MediaPipe
    handedness by the pipeline; pass it to force a side. Smoothing stays in
    perception, never in the solver path.
    """
    from src.perception.realsense_source import RealSenseHandSource
    from src.perception.pipeline import HandPipeline

    source = RealSenseHandSource()
    pipeline = HandPipeline(
        source,
        smoothing=smoothing,
        one_euro=_one_euro_config(one_euro_min_cutoff, one_euro_beta, one_euro_d_cutoff),
        image_mirrored=image_mirrored,
        force_side=side,
    )
    model = L20VizModel(side or "right", gui=gui)
    adjust = _compose_adjust(
        _thumb_adjuster(thumb_gain, thumb_cross_gain, thumb_assist_smooth),
        _little_abd_adjuster(little_abd_gain),
        _thumb_grasp_adjuster(thumb_grasp_gain),
        _thumb_base_assist_adjuster(thumb_base_assist_gain, thumb_assist_smooth),
        _thumb_orient_adjuster(thumb_orient_gain),
        _thumb_tip_adjuster(thumb_tip_gain),
        _thumb_only_adjuster(thumb_only),
    )
    cv2 = None
    if show_camera:  # optional camera/overlay window beside the sim
        try:
            import cv2  # noqa: F811
        except ImportError:
            cv2 = None

    def _stream() -> Iterator[Tuple[str, np.ndarray, float]]:
        n = 0
        for det in source:
            pf = pipeline.process(det)
            if pf is None:
                if not _camera_preview(cv2, source, "RealSense camera", None):
                    break
                continue
            # A live side flip needs a fresh model; Stage 1 mirrors one hand.
            if pf.side != model.side:
                model.side = pf.side  # render still works; informational
            if not _camera_preview(cv2, source, "RealSense camera", pf):
                break
            yield pf.side, pf.landmarks, pf.t
            n += 1
            if max_frames is not None and n >= max_frames:
                break

    try:
        records = drive(model, _stream(), use_filter=use_filter, dt=dt,
                        candidate_adjust=adjust,
                        debug_callback=_match_debug(debug_match, debug_match_period))
    finally:
        source.close()
        model.close()
        if cv2 is not None:
            cv2.destroyAllWindows()
    return records


# --------------------------------------------------------------------------- #
# Live webcam mirror (monocular RGB -- plumbing only, LOW depth confidence)
# --------------------------------------------------------------------------- #
def run_webcam(camera_index: int = 0, side: Optional[str] = None, *,
               gui: bool = True, use_filter: bool = True,
               image_mirrored: bool = False, fps: float = 30.0,
               show_camera: bool = False, dt: float = DEFAULT_DT,
               max_frames: Optional[int] = None,
               smoothing: bool = True,
               one_euro_min_cutoff: float = 1.5,
               one_euro_beta: float = 0.05,
               one_euro_d_cutoff: float = 1.0,
               fingertip_extend: float = 0.0,
               fingertip_lateral: float = 0.0,
               fingertip_straighten: float = 0.0,
               thumb_gain: float = 1.0, thumb_cross_gain: float = 1.0,
               thumb_assist_smooth: float = 0.0,
               little_abd_gain: float = 1.0,
               thumb_grasp_gain: float = 0.0,
               thumb_base_assist_gain: float = 0.0,
               thumb_tip_gain: float = 1.0,
               thumb_orient_gain: float = 0.0,
               thumb_only: bool = False,
               debug_match: bool = False,
               debug_match_period: int = 15) -> List[dict]:
    """Live webcam mirror: OpenCV camera -> MediaPipe -> viz loop.

    This is the live-camera analogue of ``run_video``: it uses MediaPipe's
    monocular world landmarks, so depth confidence is LOW. It is useful for a
    regular USB webcam when no RealSense is connected, but RealSense remains the
    higher-confidence RGB-D path for depth-sensitive retarget validation.
    """
    from src.perception.mediapipe_source import MediaPipeHandSource
    from src.perception.pipeline import HandPipeline

    source = MediaPipeHandSource(camera_index=camera_index, fps=fps,
                                 fingertip_extend=fingertip_extend,
                                 fingertip_lateral=fingertip_lateral,
                                 fingertip_straighten=fingertip_straighten)
    pipeline = HandPipeline(
        source,
        smoothing=smoothing,
        one_euro=_one_euro_config(one_euro_min_cutoff, one_euro_beta, one_euro_d_cutoff),
        image_mirrored=image_mirrored,
        force_side=side,
    )
    model = L20VizModel(side or "right", gui=gui)
    adjust = _compose_adjust(
        _thumb_adjuster(thumb_gain, thumb_cross_gain, thumb_assist_smooth),
        _little_abd_adjuster(little_abd_gain),
        _thumb_grasp_adjuster(thumb_grasp_gain),
        _thumb_base_assist_adjuster(thumb_base_assist_gain, thumb_assist_smooth),
        _thumb_orient_adjuster(thumb_orient_gain),
        _thumb_tip_adjuster(thumb_tip_gain),
        _thumb_only_adjuster(thumb_only),
    )
    cv2 = None
    if show_camera:
        import cv2  # noqa: F811

    def _stream() -> Iterator[Tuple[str, np.ndarray, float]]:
        n = 0
        for det in source:
            pf = pipeline.process(det)
            if pf is None:
                if not _camera_preview(cv2, source, "Webcam camera", None):
                    break
                continue
            if pf.side != model.side:
                model.side = pf.side  # render still works; informational
            if not _camera_preview(cv2, source, "Webcam camera", pf):
                break
            yield pf.side, pf.landmarks, pf.t
            n += 1
            if max_frames is not None and n >= max_frames:
                break

    try:
        records = drive(model, _stream(), use_filter=use_filter, dt=dt,
                        candidate_adjust=adjust,
                        debug_callback=_match_debug(debug_match, debug_match_period))
    finally:
        source.close()
        model.close()
        if cv2 is not None:
            cv2.destroyAllWindows()
    return records


# --------------------------------------------------------------------------- #
# Recorded-video mirror (monocular RGB — plumbing only, LOW depth confidence)
# --------------------------------------------------------------------------- #
def run_video(video_path: str, side: Optional[str] = None, *, gui: bool = True,
              use_filter: bool = True, image_mirrored: bool = False,
              playback_rate: float = 1.0, fps: Optional[float] = None,
              realtime: Optional[bool] = None, dt: float = DEFAULT_DT,
              max_frames: Optional[int] = None,
              smoothing: bool = True,
              one_euro_min_cutoff: float = 1.5,
              one_euro_beta: float = 0.05,
              one_euro_d_cutoff: float = 1.0,
              fingertip_extend: float = 0.0,
              fingertip_lateral: float = 0.0,
              fingertip_straighten: float = 0.0,
              thumb_gain: float = 1.0, thumb_cross_gain: float = 1.0,
              thumb_assist_smooth: float = 0.0,
              little_abd_gain: float = 1.0,
              thumb_grasp_gain: float = 0.0,
              thumb_base_assist_gain: float = 0.0,
              thumb_tip_gain: float = 1.0,
              thumb_orient_gain: float = 0.0,
              thumb_only: bool = False,
              debug_match: bool = False,
              debug_match_period: int = 15) -> List[dict]:
    """Recorded-video mirror: a hand clip -> MediaPipe -> pipeline -> viz loop.

    Same loop as the live/replay paths, but the source is a monocular RGB video
    file (``VideoHandSource``), so depth confidence is LOW — this exercises the
    end-to-end plumbing (real hand motion through to the sim hand), not depth or
    retarget accuracy. ``side`` is resolved per-frame from MediaPipe handedness
    unless forced; if the sim hand mirrors the wrong way on a selfie-recorded
    clip, set ``image_mirrored`` (see handedness.to_l20_side). Honors the clip's
    native FPS; ``playback_rate`` scales wall-clock pacing only. EOF stops cleanly.
    """
    from src.perception.video_source import VideoHandSource
    from src.perception.pipeline import HandPipeline

    if realtime is None:
        realtime = gui
    source = VideoHandSource(video_path, fps=fps, playback_rate=playback_rate,
                             fingertip_extend=fingertip_extend,
                             fingertip_lateral=fingertip_lateral,
                             fingertip_straighten=fingertip_straighten)
    pipeline = HandPipeline(
        source,
        smoothing=smoothing,
        one_euro=_one_euro_config(one_euro_min_cutoff, one_euro_beta, one_euro_d_cutoff),
        image_mirrored=image_mirrored,
        force_side=side,
    )
    model = L20VizModel(side or "right", gui=gui)
    adjust = _compose_adjust(
        _thumb_adjuster(thumb_gain, thumb_cross_gain, thumb_assist_smooth),
        _little_abd_adjuster(little_abd_gain),
        _thumb_grasp_adjuster(thumb_grasp_gain),
        _thumb_base_assist_adjuster(thumb_base_assist_gain, thumb_assist_smooth),
        _thumb_orient_adjuster(thumb_orient_gain),
        _thumb_tip_adjuster(thumb_tip_gain),
        _thumb_only_adjuster(thumb_only),
    )
    period = source.frame_period

    def _stream() -> Iterator[Tuple[str, np.ndarray, float]]:
        n = 0
        for pf in pipeline.run():
            if pf.side != model.side:  # render still works; informational
                model.side = pf.side
            yield pf.side, pf.landmarks, pf.t
            if realtime:
                time.sleep(period)
            n += 1
            if max_frames is not None and n >= max_frames:
                break

    try:
        records = drive(model, _stream(), use_filter=use_filter, dt=dt,
                        candidate_adjust=adjust,
                        debug_callback=_match_debug(debug_match, debug_match_period))
    finally:
        source.close()
        model.close()
    return records


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Stage 1 L20 teleop visualizer (live camera or camera-free).")
    ap.add_argument("--source", choices=("replay", "realsense", "webcam", "video"),
                    default=None,
                    help="input source (default: realsense live; --camera-free => replay)")
    ap.add_argument("--camera-index", type=int, default=0,
                    help="webcam: OpenCV camera index (default: 0)")
    ap.add_argument("--video-path", default=None,
                    help="video file path (required for --source video)")
    ap.add_argument("--playback-rate", type=float, default=1.0,
                    help="video: wall-clock playback speed (1.0 = native FPS)")
    ap.add_argument("--image-mirrored", action="store_true",
                    help="input frame is a selfie/mirrored view (handedness swap)")
    ap.add_argument("--fingertip-extend", default="0.0",
                    help=("webcam/video: extend MediaPipe fingertips along distal bones; "
                          "scalar or thumb,index,middle,ring,little"))
    ap.add_argument("--fingertip-lateral", default="0.0",
                    help=("webcam/video: shift MediaPipe fingertips sideways; "
                          "scalar or thumb,index,middle,ring,little, + toward index side"))
    ap.add_argument("--fingertip-straighten", default="0.0",
                    help=("webcam/video: blend fingertips toward PIP->DIP continuation; "
                          "scalar or thumb,index,middle,ring,little"))
    ap.add_argument("--thumb-gain", type=float, default=1.0,
                    help="multiply thumb actuators after retargeting, before safety")
    ap.add_argument("--thumb-cross-gain", type=float, default=1.0,
                    help="push thumb farther across palm near four-finger MCP roots")
    ap.add_argument("--thumb-assist-smooth", type=float, default=0.0,
                    help="smooth thumb cross/contact assist over time, 0..0.95")
    ap.add_argument("--little-abd-gain", type=float, default=1.0,
                    help="scale little-finger side orientation q9; 1.0 restores raw retarget")
    ap.add_argument("--thumb-grasp-gain", type=float, default=0.0,
                    help="sim-only thumb opposition/tip grasp assist, 0..1")
    ap.add_argument("--thumb-base-assist-gain", type=float, default=0.0,
                    help="sim-only: add thumb q0 reach near index/middle fingertips")
    ap.add_argument("--thumb-tip-gain", type=float, default=1.0,
                    help="sim-only: scale only thumb tip curl q15 after assists")
    ap.add_argument("--thumb-orient-gain", type=float, default=0.0,
                    help="sim-only: use distal thumb direction to boost opposition")
    ap.add_argument("--thumb-only", action="store_true",
                    help="sim-only: freeze non-thumb joints open for thumb-map tuning")
    ap.add_argument("--camera-free", action="store_true",
                    help="replay the committed synthetic_openclose instead of a camera")
    ap.add_argument("--side", choices=("right", "left"), default="right",
                    help="hand side (camera-free; live/video resolve from handedness)")
    ap.add_argument("--no-filter", action="store_true",
                    help="bypass safety.filter (raw retarget output)")
    ap.add_argument("--no-smoothing", action="store_true",
                    help="disable perception-side one-euro landmark smoothing")
    ap.add_argument("--one-euro-min-cutoff", type=float, default=1.5,
                    help="perception One Euro min cutoff in Hz; lower is smoother but laggier")
    ap.add_argument("--one-euro-beta", type=float, default=0.05,
                    help="perception One Euro speed coefficient; higher reduces lag while moving")
    ap.add_argument("--one-euro-d-cutoff", type=float, default=1.0,
                    help="perception One Euro derivative cutoff in Hz")
    ap.add_argument("--max-frames", type=int, default=None,
                    help="stop after this many processed frames")
    ap.add_argument("--loop", action="store_true",
                    help="camera-free: repeat the sequence forever")
    ap.add_argument("--headless", action="store_true",
                    help="DIRECT (no GUI window) — for smoke runs")
    ap.add_argument("--show-camera", action="store_true",
                    help="live: also show the camera feed window")
    ap.add_argument("--debug-match", action="store_true",
                    help="print MediaPipe-vs-sim per-finger angle diagnostics")
    ap.add_argument("--debug-match-period", type=int, default=15,
                    help="print one match diagnostic every N processed frames")
    args = ap.parse_args(argv)

    gui = not args.headless
    use_filter = not args.no_filter
    smoothing = not args.no_smoothing
    source = args.source or ("replay" if args.camera_free else "realsense")
    if source == "replay":
        run_camera_free(args.side, gui=gui, use_filter=use_filter, loop=args.loop,
                        max_frames=args.max_frames, thumb_gain=args.thumb_gain,
                        thumb_cross_gain=args.thumb_cross_gain,
                        thumb_assist_smooth=args.thumb_assist_smooth,
                        little_abd_gain=args.little_abd_gain,
                        thumb_grasp_gain=args.thumb_grasp_gain,
                        thumb_base_assist_gain=args.thumb_base_assist_gain,
                        thumb_tip_gain=args.thumb_tip_gain,
                        thumb_orient_gain=args.thumb_orient_gain,
                        thumb_only=args.thumb_only,
                        debug_match=args.debug_match,
                        debug_match_period=args.debug_match_period)
    elif source == "webcam":
        run_webcam(args.camera_index, side=args.side, gui=gui, use_filter=use_filter,
                   image_mirrored=args.image_mirrored, show_camera=args.show_camera,
                   smoothing=smoothing,
                   one_euro_min_cutoff=args.one_euro_min_cutoff,
                   one_euro_beta=args.one_euro_beta,
                   one_euro_d_cutoff=args.one_euro_d_cutoff,
                   max_frames=args.max_frames, fingertip_extend=args.fingertip_extend,
                   fingertip_lateral=args.fingertip_lateral,
                   fingertip_straighten=args.fingertip_straighten,
                   thumb_gain=args.thumb_gain,
                   thumb_cross_gain=args.thumb_cross_gain,
                   thumb_assist_smooth=args.thumb_assist_smooth,
                   little_abd_gain=args.little_abd_gain,
                   thumb_grasp_gain=args.thumb_grasp_gain,
                   thumb_base_assist_gain=args.thumb_base_assist_gain,
                   thumb_tip_gain=args.thumb_tip_gain,
                   thumb_orient_gain=args.thumb_orient_gain,
                   thumb_only=args.thumb_only,
                   debug_match=args.debug_match,
                   debug_match_period=args.debug_match_period)
    elif source == "video":
        if not args.video_path:
            ap.error("--source video requires --video-path")
        # side resolved from MediaPipe handedness; --image-mirrored for selfie clips
        run_video(args.video_path, side=None, gui=gui, use_filter=use_filter,
                  image_mirrored=args.image_mirrored, playback_rate=args.playback_rate,
                  smoothing=smoothing,
                  one_euro_min_cutoff=args.one_euro_min_cutoff,
                  one_euro_beta=args.one_euro_beta,
                  one_euro_d_cutoff=args.one_euro_d_cutoff,
                  max_frames=args.max_frames, fingertip_extend=args.fingertip_extend,
                  fingertip_lateral=args.fingertip_lateral,
                  fingertip_straighten=args.fingertip_straighten,
                  thumb_gain=args.thumb_gain,
                  thumb_cross_gain=args.thumb_cross_gain,
                  thumb_assist_smooth=args.thumb_assist_smooth,
                  little_abd_gain=args.little_abd_gain,
                  thumb_grasp_gain=args.thumb_grasp_gain,
                  thumb_base_assist_gain=args.thumb_base_assist_gain,
                  thumb_tip_gain=args.thumb_tip_gain,
                  thumb_orient_gain=args.thumb_orient_gain,
                  thumb_only=args.thumb_only,
                  debug_match=args.debug_match,
                  debug_match_period=args.debug_match_period)
    else:
        run_live(args.side, gui=gui, use_filter=use_filter,
                 image_mirrored=args.image_mirrored, show_camera=args.show_camera,
                 smoothing=smoothing,
                 one_euro_min_cutoff=args.one_euro_min_cutoff,
                 one_euro_beta=args.one_euro_beta,
                 one_euro_d_cutoff=args.one_euro_d_cutoff,
                 max_frames=args.max_frames, thumb_gain=args.thumb_gain,
                 thumb_cross_gain=args.thumb_cross_gain,
                 thumb_assist_smooth=args.thumb_assist_smooth,
                 little_abd_gain=args.little_abd_gain,
                 thumb_grasp_gain=args.thumb_grasp_gain,
                 thumb_base_assist_gain=args.thumb_base_assist_gain,
                 thumb_tip_gain=args.thumb_tip_gain,
                 thumb_orient_gain=args.thumb_orient_gain,
                 thumb_only=args.thumb_only,
                 debug_match=args.debug_match,
                 debug_match_period=args.debug_match_period)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
