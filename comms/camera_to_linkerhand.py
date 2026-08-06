"""Camera -> retarget -> safety -> LinkerHand ROS2 bridge.

Default is dry-run. Real motion requires BOTH:
  1. ``--enable-motion``
  2. a human-set ``HW_ENABLE_TOKEN`` environment variable

The LinkerHand ROS2 SDK driver subscribes to ``/cb_<side>_hand_control_cmd`` as
``sensor_msgs/JointState`` with 20 position values in SDK range units (0..255).
This module keeps the rest of the repo in radians, then maps the final safe L20
command into that SDK range immediately before publishing.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from typing import Iterator, List, Optional, Tuple

import numpy as np

from src.viz.core import DEFAULT_DT, drive


ACTIVE_IDX = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 16, 17, 18, 19)
RESERVED_IDX = (11, 12, 13, 14)
OPEN_RANGE_L20 = [
    255, 255, 255, 255, 255,
    255, 10, 100, 180, 240,
    245, 255, 255, 255, 255,
    255, 255, 255, 255, 255,
]
OPEN_RANGE_G20 = [
    255, 255, 255, 255, 255,
    255, 193, 148, 105, 42,
    245, 255, 255, 255, 255,
    255, 255, 255, 255, 255,
]
OPEN_RANGES = {
    "l20": OPEN_RANGE_L20,
    "g20": OPEN_RANGE_G20,
}
OPEN_RANGE = OPEN_RANGE_L20
G20_ACTION16_TO_CMD20 = (
    0, 1, 2, 3, 4,
    5, 6, 7, 8, 9,
    10,
    15, 16, 17, 18, 19,
)
G20_SIM_LOWER16 = np.asarray([
    0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, -0.17, -0.17, -0.17, -0.17,
    0.0,
    0.0, 0.0, 0.0, 0.0, 0.0,
], dtype=float)
G20_SIM_UPPER16 = np.asarray([
    0.79, 1.4, 1.4, 1.4, 1.4,
    1.4, 0.17, 0.17, 0.17, 0.17,
    1.22,
    1.05, 1.57, 1.57, 1.57, 1.57,
], dtype=float)
G20_ROLL_ACTION_IDS = (6, 7, 8, 9)
BASE_IDX = (0, 1, 2, 3, 4)
SPREAD_IDX = (5, 6, 7, 8, 9, 10)
TIP_IDX = (15, 16, 17, 18, 19)
NONTHUMB_BASE_IDX = (1, 2, 3, 4)
NONTHUMB_SPREAD_IDX = (6, 7, 8, 9)
NONTHUMB_TIP_IDX = (16, 17, 18, 19)
THUMB_COLLISION_IDX = (5, 10)
L20_NAMES = [
    "thumb_base",
    "index_base",
    "middle_base",
    "ring_base",
    "little_base",
    "thumb_abduction",
    "index_abduction",
    "middle_abduction",
    "ring_abduction",
    "little_abduction",
    "thumb_roll",
    "reserved_11",
    "reserved_12",
    "reserved_13",
    "reserved_14",
    "thumb_tip",
    "index_tip",
    "middle_tip",
    "ring_tip",
    "little_tip",
]

# LinkerHand SDK L20 arc<->range constants, copied from the installed SDK's
# LinkerHand/utils/mapping.py. ``direct == -1`` means 255 is the minimum-open end.
L20_MIN = {
    "left":  [0, 0, 0, 0, 0, -0.297, -0.26, -0.26, -0.26, -0.26, 0.122, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "right": [0, 0, 0, 0, 0, -0.297, -0.26, -0.26, -0.26, -0.26, 0,     0, 0, 0, 0, 0, 0, 0, 0, 0],
}
L20_MAX = {
    "left":  [0.87, 1.4, 1.4, 1.4, 1.4, 0.683, 0.26, 0.26, 0.26, 0.26, 1.78, 0, 0, 0, 0, 1.29, 1.08, 1.08, 1.08, 1.08],
    "right": [0.87, 1.4, 1.4, 1.4, 1.4, 0.683, 0.26, 0.26, 0.26, 0.26, 1.78, 0, 0, 0, 0, 1.29, 1.08, 1.08, 1.08, 1.08],
}
L20_DIRECT = {
    "left":  [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, -1, -1, -1, -1, -1],
    "right": [-1, -1, -1, -1, -1, -1,  0,  0,  0,  0, -1, 0, 0, 0, 0, -1, -1, -1, -1, -1],
}


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _scale(value: float, in_lo: float, in_hi: float, out_lo: float, out_hi: float) -> float:
    if abs(in_hi - in_lo) < 1e-12:
        return out_lo
    alpha = (value - in_lo) / (in_hi - in_lo)
    return out_lo + alpha * (out_hi - out_lo)


def _parse_four_floats(text: str, *, name: str) -> Tuple[float, float, float, float]:
    parts = [p.strip() for p in str(text).split(",") if p.strip()]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError(f"{name} must contain four comma-separated values")
    try:
        return tuple(float(p) for p in parts)  # type: ignore[return-value]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{name} contains a non-number") from exc


def l20_radians_to_sdk_range(joint_rad, side: str,
                             open_range: Optional[List[int]] = None) -> List[int]:
    """Map a 20-vector in radians to SDK range units (0..255)."""
    if side not in L20_MIN:
        raise ValueError(f"side must be left/right, got {side!r}")
    q = np.asarray(joint_rad, dtype=float).reshape(-1)
    if q.shape[0] != 20:
        raise ValueError(f"joint_rad must have 20 entries, got {q.shape[0]}")

    out = list(open_range or OPEN_RANGE_L20)
    mn, mx, direct = L20_MIN[side], L20_MAX[side], L20_DIRECT[side]
    for i in ACTIVE_IDX:
        val = _clamp(float(q[i]), mn[i], mx[i])
        if direct[i] == -1:
            out[i] = int(round(_scale(val, mn[i], mx[i], 255, 0)))
        else:
            out[i] = int(round(_scale(val, mn[i], mx[i], 0, 255)))
        out[i] = max(0, min(255, out[i]))
    for i in RESERVED_IDX:
        out[i] = (open_range or OPEN_RANGE_L20)[i]
    return out


def g20_sim_radians_to_sdk_range(joint_rad, *, open_range: List[int],
                                 roll_range_ticks: float,
                                 base_gain: float = 1.0,
                                 base_gains: Tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
                                 spread_gain: float = 1.0,
                                 spread_signs: Tuple[float, float, float, float] = (-1.0, -1.0, -1.0, -1.0),
                                 tip_gain: float = 1.0,
                                 tip_gains: Tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
                                 thumb_base_gain: float = 1.0,
                                 thumb_abd_gain: float = 1.0,
                                 thumb_roll_gain: float = 1.0,
                                 thumb_tip_gain: float = 1.0,
                                 thumb_base_offset: int = 0,
                                 thumb_abd_offset: int = 0,
                                 thumb_roll_offset: int = 0,
                                 thumb_tip_offset: int = 0) -> List[int]:
    """Map the same radians shown in PyBullet sim to the G20 palm-touch SDK range.

    This mirrors the SDK-side HORA helper ``sim_rad_to_active_range`` so the
    hardware bridge follows the visualized pose instead of the L20 SDK arc table.
    """
    q = np.asarray(joint_rad, dtype=float).reshape(-1)
    if q.shape[0] != 20:
        raise ValueError(f"joint_rad must have 20 entries, got {q.shape[0]}")

    sim = np.asarray([q[i] for i in G20_ACTION16_TO_CMD20], dtype=float)
    gains = np.ones_like(sim)
    for idx in BASE_IDX:
        gains[G20_ACTION16_TO_CMD20.index(idx)] = float(base_gain)
    for idx, gain in zip(NONTHUMB_BASE_IDX, base_gains):
        gains[G20_ACTION16_TO_CMD20.index(idx)] = float(base_gain) * float(gain)
    for idx in SPREAD_IDX:
        gains[G20_ACTION16_TO_CMD20.index(idx)] = float(spread_gain)
    for idx in TIP_IDX:
        gains[G20_ACTION16_TO_CMD20.index(idx)] = float(tip_gain)
    for idx, gain in zip(NONTHUMB_TIP_IDX, tip_gains):
        gains[G20_ACTION16_TO_CMD20.index(idx)] = float(tip_gain) * float(gain)
    gains[G20_ACTION16_TO_CMD20.index(0)] = float(thumb_base_gain)
    gains[G20_ACTION16_TO_CMD20.index(5)] = float(thumb_abd_gain)
    gains[G20_ACTION16_TO_CMD20.index(10)] = float(thumb_roll_gain)
    gains[G20_ACTION16_TO_CMD20.index(15)] = float(thumb_tip_gain)
    sim = sim * gains
    for idx, sign in zip(NONTHUMB_SPREAD_IDX, spread_signs):
        sim[G20_ACTION16_TO_CMD20.index(idx)] *= float(sign)
    sim = np.clip(sim, G20_SIM_LOWER16, G20_SIM_UPPER16)

    active_open = np.asarray([open_range[i] for i in G20_ACTION16_TO_CMD20], dtype=float)
    active = active_open - (
        (sim - G20_SIM_LOWER16)
        / (G20_SIM_UPPER16 - G20_SIM_LOWER16)
        * active_open
    )
    for action_i in G20_ROLL_ACTION_IDS:
        active[action_i] = (
            active_open[action_i]
            + sim[action_i] / G20_SIM_UPPER16[action_i] * max(1.0, float(roll_range_ticks))
        )

    out = list(open_range)
    for value, idx in zip(active, G20_ACTION16_TO_CMD20):
        out[idx] = max(0, min(255, int(round(float(value)))))
    out[0] = max(0, min(255, out[0] + int(thumb_base_offset)))
    out[5] = max(0, min(255, out[5] + int(thumb_abd_offset)))
    out[10] = max(0, min(255, out[10] + int(thumb_roll_offset)))
    out[15] = max(0, min(255, out[15] + int(thumb_tip_offset)))
    for i in RESERVED_IDX:
        out[i] = open_range[i]
    return out


def _limit_range_step(target: List[int], prev: Optional[List[int]], max_step: int) -> List[int]:
    if prev is None or max_step <= 0:
        return target
    out = list(target)
    for i in ACTIVE_IDX:
        delta = out[i] - prev[i]
        if delta > max_step:
            out[i] = prev[i] + max_step
        elif delta < -max_step:
            out[i] = prev[i] - max_step
    return out


def _closure_from_open(pose: List[int], open_range: List[int], indices: Tuple[int, ...]) -> float:
    vals = []
    for idx in indices:
        span = max(1.0, float(open_range[idx]))
        vals.append(_clamp((float(open_range[idx]) - float(pose[idx])) / span, 0.0, 1.0))
    return max(vals) if vals else 0.0


def apply_nonthumb_close_deadzone(
    pose: List[int],
    open_range: List[int],
    *,
    deadzone: int,
) -> List[int]:
    """Suppress small accidental four-finger closure in SDK range space."""
    dz = max(0, int(deadzone))
    if dz <= 0:
        return list(pose)
    out = list(pose)
    for idx in NONTHUMB_BASE_IDX + NONTHUMB_TIP_IDX:
        open_v = int(open_range[idx])
        delta = open_v - int(out[idx])
        if delta <= 0:
            out[idx] = open_v
        elif delta <= dz:
            out[idx] = open_v
        else:
            out[idx] = max(0, min(255, out[idx] + dz))
    for i in RESERVED_IDX:
        out[i] = open_range[i]
    return out


def apply_hardware_collision_guard(
    pose: List[int],
    open_range: List[int],
    *,
    enabled: bool,
    thumb_safe_mode: str,
    max_thumb_delta: int,
    max_thumb_abd_delta: int,
    max_thumb_base_delta: Optional[int],
    max_spread_delta: int,
    spread_close_threshold: float,
    spread_recenter_gain: float,
    min_spread_gap: int,
) -> List[int]:
    """Conservative G20 range-space guard for thumb/finger fighting.

    This is a hardware-output guard, separate from the sim/L20 radian safety
    filter. It keeps the G20 palm-touch thumb rotation and four-finger spread
    channels in a stable corridor before publishing to the real hand.
    """
    if not enabled:
        return list(pose)

    out = list(pose)

    if thumb_safe_mode == "open":
        for idx in THUMB_COLLISION_IDX:
            out[idx] = open_range[idx]
    elif thumb_safe_mode == "limited":
        limits = {
            0: max(0, int(max_thumb_base_delta)) if max_thumb_base_delta is not None else None,
            5: max(0, int(max_thumb_abd_delta)),
            10: max(0, int(max_thumb_delta)),
        }
        for idx in (0,) + THUMB_COLLISION_IDX:
            limit = limits.get(idx, max(0, int(max_thumb_delta)))
            if limit is None:
                continue
            out[idx] = max(open_range[idx] - limit, min(open_range[idx] + limit, out[idx]))

    closure = _closure_from_open(out, open_range, NONTHUMB_BASE_IDX + NONTHUMB_TIP_IDX)
    if closure > spread_close_threshold:
        alpha = _clamp((closure - spread_close_threshold) / max(1e-6, 1.0 - spread_close_threshold), 0.0, 1.0)
        alpha *= _clamp(spread_recenter_gain, 0.0, 1.0)
        for idx in NONTHUMB_SPREAD_IDX:
            out[idx] = int(round(out[idx] * (1.0 - alpha) + open_range[idx] * alpha))

    spread_limit = max(0, int(max_spread_delta))
    for idx in NONTHUMB_SPREAD_IDX:
        out[idx] = max(open_range[idx] - spread_limit, min(open_range[idx] + spread_limit, out[idx]))

    gap = max(0, int(min_spread_gap))
    # Preserve the G20 palm-touch open ordering: index >= middle >= ring >= little.
    for left, right in zip(NONTHUMB_SPREAD_IDX, NONTHUMB_SPREAD_IDX[1:]):
        if out[left] < out[right] + gap:
            out[left] = min(255, out[right] + gap)
    for right, left in zip(reversed(NONTHUMB_SPREAD_IDX[1:]), reversed(NONTHUMB_SPREAD_IDX[:-1])):
        if out[right] > out[left] - gap:
            out[right] = max(0, out[left] - gap)

    for i in RESERVED_IDX:
        out[i] = open_range[i]
    return [max(0, min(255, int(round(v)))) for v in out]


def apply_thumb_index_guard(
    pose: List[int],
    open_range: List[int],
    *,
    enabled: bool,
    threshold: float,
    release_ticks: int,
) -> List[int]:
    """Extra range-space guard for the high-risk thumb/index pinch path.

    G20 range values close as they move below ``open_range`` for flexion. If the
    thumb is rotated into the palm while the index base/tip are also closing,
    the two links can fight. In that case, open the index a little first and
    softly cap thumb side-swing.
    """
    if not enabled:
        return list(pose)

    out = list(pose)
    thumb_closure = _closure_from_open(out, open_range, (0, 5, 10, 15))
    index_closure = _closure_from_open(out, open_range, (1, 16))
    risk = min(thumb_closure, index_closure)
    if risk <= float(threshold):
        return out

    alpha = _clamp((risk - float(threshold)) / max(1e-6, 1.0 - float(threshold)), 0.0, 1.0)
    release = int(round(max(0, int(release_ticks)) * alpha))
    for idx in (1, 16):
        if out[idx] < open_range[idx]:
            out[idx] = min(open_range[idx], out[idx] + release)

    for i in RESERVED_IDX:
        out[i] = open_range[i]
    return [max(0, min(255, int(round(v)))) for v in out]


def _hardware_candidate_adjuster(
    thumb_gain: float,
    *,
    landmark_thumb: bool,
    landmark_thumb_gain: float,
    landmark_thumb_reach_gain: float,
    landmark_spread: bool,
    landmark_spread_gain: float,
    landmark_spread_limit: float,
    landmark_spread_calibration_frames: int,
):
    thumb_gain = float(thumb_gain)
    landmark_thumb_gain = float(landmark_thumb_gain)
    landmark_thumb_reach_gain = max(0.0, float(landmark_thumb_reach_gain))
    landmark_spread_gain = float(landmark_spread_gain)
    landmark_spread_limit = float(landmark_spread_limit)
    if (
        abs(thumb_gain - 1.0) < 1e-12
        and (not landmark_thumb or landmark_thumb_gain <= 0.0)
        and (not landmark_spread or landmark_spread_gain <= 0.0 or landmark_spread_limit <= 0.0)
    ):
        return None
    from src.finger_retarget.constants import CONSTANTS

    baseline_samples: List[List[float]] = []
    baseline_angles: Optional[np.ndarray] = None

    def _adjust(candidate: dict, *, landmarks=None) -> dict:
        nonlocal baseline_angles
        side = candidate.get("side", "right")
        thumb = CONSTANTS[side]["thumb"]
        limits = {int(idx): tuple(lim) for idx, _axis, lim in thumb["base_axes"]}
        limits[int(thumb["tip_idx"])] = tuple(thumb["tip_limit"])
        q = list(candidate["joint_rad"])
        if abs(thumb_gain - 1.0) >= 1e-12:
            for idx, (lo, hi) in limits.items():
                q[idx] = _clamp(q[idx] * thumb_gain, lo, hi)

        if landmark_thumb and landmark_thumb_gain > 0.0 and landmarks is not None:
            lm = np.asarray(landmarks, dtype=float)
            if lm.shape == (21, 3) and np.all(np.isfinite(lm)):
                roots = lm[[5, 9, 13, 17]]
                root_line = roots[-1] - roots[0]
                line_len2 = float(root_line @ root_line)
                if line_len2 > 1e-10:
                    def _root_t(point) -> float:
                        return _clamp(float(((point - roots[0]) @ root_line) / line_len2), 0.0, 1.0)

                    # Use the reliable front thumb chain (2/3/4), not the CMC/wrist
                    # point, so visible distal thumb rotation creates real opposition.
                    raw_t2 = _root_t(lm[2])
                    raw_t3 = _root_t(lm[3])
                    raw_t4 = _root_t(lm[4])
                    t2 = _clamp(raw_t2 * landmark_thumb_reach_gain, 0.0, 1.0)
                    t3 = _clamp(raw_t3 * landmark_thumb_reach_gain, 0.0, 1.0)
                    t4 = _clamp(raw_t4 * landmark_thumb_reach_gain, 0.0, 1.0)
                    front_sweep = _clamp(0.10 * t2 + 0.25 * t3 + 0.65 * t4, 0.0, 1.0)
                    # Direction of the visible distal thumb chain. This changes
                    # even when the CMC/root landmark barely moves in the camera.
                    front_dir = _clamp((t4 - t2) / 0.60, 0.0, 1.0)
                    opposition = _clamp(0.78 * front_sweep + 0.22 * front_dir, 0.0, 1.0)
                    opposition = opposition ** 0.65
                    side_swing = _clamp(0.35 * front_sweep + 0.65 * front_dir, 0.0, 1.0)
                    side_swing = side_swing ** 0.55
                    v23 = lm[3] - lm[2]
                    v34 = lm[4] - lm[3]
                    n23 = float(np.linalg.norm(v23))
                    n34 = float(np.linalg.norm(v34))
                    distal_bend = 0.0
                    if n23 > 1e-9 and n34 > 1e-9:
                        cos_tip = float((v23 @ v34) / (n23 * n34))
                        cos_tip = _clamp(cos_tip, -1.0, 1.0)
                        distal_bend = _clamp(float(np.arccos(cos_tip)) / 0.9, 0.0, 1.0)
                        distal_bend = distal_bend ** 0.65

                    closest = roots[0] + raw_t4 * root_line
                    palm_width = float(np.sqrt(line_len2))
                    line_dist = float(np.linalg.norm(lm[4] - closest))
                    contact = _clamp((0.72 * palm_width - line_dist) / (0.45 * palm_width), 0.0, 1.0)
                    # Do not add landmark-thumb side swing in a normal open hand.
                    # It should help only near pinch/grasp contact, otherwise the
                    # hardware thumb never returns fully open.
                    contact = _clamp((contact - 0.18) / 0.82, 0.0, 1.0)
                    blend = _clamp(landmark_thumb_gain * contact, 0.0, 1.0)

                    abd_idx = int(thumb["abd_idx"])
                    base_idx = int(thumb["base_idx"])
                    tip_idx = int(thumb["tip_idx"])
                    deltas = {
                        # The normal retargeter can leave q0 at zero when the
                        # front thumb chain is foreshortened in a monocular
                        # image. Add contact-gated base flexion so a visible
                        # thumb press drives the G20 q0 channel below 255.  The
                        # contact gate above still returns an open hand to q0=0
                        # instead of imposing a fixed thumb posture.
                        base_idx: 0.48 * side_swing,
                        abd_idx: 0.72 * side_swing,
                        tip_idx: 0.08 * opposition + 0.30 * distal_bend,
                    }
                    for idx, delta in deltas.items():
                        lo, hi = limits[idx]
                        q[idx] = _clamp(q[idx] + blend * delta, lo, hi)

        if landmark_spread and landmark_spread_gain > 0.0 and landmark_spread_limit > 0.0 and landmarks is not None:
            lm = np.asarray(landmarks, dtype=float)
            if lm.shape == (21, 3) and np.all(np.isfinite(lm)):
                angles = []
                for mcp, pip in ((5, 6), (9, 10), (13, 14), (17, 18)):
                    v = lm[pip] - lm[mcp]
                    n = float(np.linalg.norm(v))
                    if n < 1e-9:
                        break
                    v = v / n
                    angles.append(float(np.arctan2(v[1], max(1e-6, v[2]))))
                if len(angles) == 4:
                    centered = np.asarray(angles, dtype=float)
                    centered -= float(centered.mean())
                    if baseline_angles is None:
                        baseline_samples.append([float(v) for v in centered])
                        if len(baseline_samples) >= max(1, int(landmark_spread_calibration_frames)):
                            baseline_angles = np.asarray(baseline_samples, dtype=float).mean(axis=0)
                            print("[hardware] landmark spread baseline "
                                  f"{[round(float(v), 4) for v in baseline_angles]}",
                                  flush=True)
                        for idx in NONTHUMB_SPREAD_IDX:
                            q[idx] = 0.0
                        out = dict(candidate)
                        out["joint_rad"] = q
                        return out

                    spread_limit = min(0.17, max(0.0, landmark_spread_limit))
                    for idx, delta in zip(NONTHUMB_SPREAD_IDX, centered - baseline_angles):
                        q[idx] = _clamp(float(delta) * landmark_spread_gain,
                                        -spread_limit, spread_limit)

        out = dict(candidate)
        out["joint_rad"] = q
        return out

    return _adjust


_HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
)


def _handle_thumb_key(key: int, thumb_keys) -> bool:
    """Handle all four G20 thumb channels and the Q return key."""
    if thumb_keys is None:
        return False
    if key in (ord("w"), ord("W")):
        thumb_keys.start_manual_control()
        thumb_keys.q0.adjust(+1)
        return True
    if key in (ord("s"), ord("S")):
        thumb_keys.start_manual_control()
        thumb_keys.q0.adjust(-1)
        return True
    if key in (ord("l"), ord("L")):
        thumb_keys.start_manual_control()
        thumb_keys.abd.adjust(+1)
        return True
    if key in (ord("j"), ord("J")):
        thumb_keys.start_manual_control()
        thumb_keys.abd.adjust(-1)
        return True
    if key in (ord("i"), ord("I")):
        thumb_keys.start_manual_control()
        thumb_keys.roll.adjust(+1)
        return True
    if key in (ord("k"), ord("K")):
        thumb_keys.start_manual_control()
        thumb_keys.roll.adjust(-1)
        return True
    if key in (ord("d"), ord("D")):
        thumb_keys.start_manual_control()
        thumb_keys.tip.adjust(+1)
        return True
    if key in (ord("a"), ord("A")):
        thumb_keys.start_manual_control()
        thumb_keys.tip.adjust(-1)
        return True
    if key in (ord("q"), ord("Q")):
        thumb_keys.toggle_mediapipe_thumb()
        return True
    if key in (ord("r"), ord("R")):
        thumb_keys.start_return_to_open()
        return True
    return False


def _camera_preview(cv2, source, window: str, pf, gate=None, thumb_keys=None) -> bool:
    if cv2 is None:
        return True
    frame = getattr(source, "last_frame_bgr", None)
    if frame is None:
        return True
    view = frame.copy()
    pts_px = getattr(source, "last_landmarks_px", None)
    if pts_px is not None and len(pts_px) >= 21:
        h, w = view.shape[:2]
        pts = []
        for px, py in pts_px[:21]:
            if not np.isfinite(px) or not np.isfinite(py):
                pts = []
                break
            pts.append((max(0, min(w - 1, int(round(px)))),
                        max(0, min(h - 1, int(round(py))))))
        if pts:
            for a, b in _HAND_CONNECTIONS:
                cv2.line(view, pts[a], pts[b], (0, 255, 255), 2, cv2.LINE_AA)
            for i, p in enumerate(pts):
                color = (0, 180, 255) if i in (4, 8, 12, 16, 20) else (0, 255, 0)
                cv2.circle(view, p, 4, color, -1, cv2.LINE_AA)
    text = "no hand" if pf is None else f"{pf.side}  score={pf.score:.2f}"
    cv2.putText(view, text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 255, 0), 2, cv2.LINE_AA)
    if gate is not None:
        on = gate.active
        if not on:
            if thumb_keys is not None and thumb_keys.manual_override:
                status = "STOPPED - MANUAL THUMB (space: follow)"
                color = (255, 220, 0)
            elif getattr(gate, "stop_mode", "open") == "hold":
                status = "STOPPED - HOLDING (space: start)"
                color = (0, 165, 255)
            else:
                status = "STOPPED - reset (space: start)"
                color = (0, 165, 255)
        elif gate.hand_tracking:
            status = "REC + FOLLOWING (space: stop)"
            color = (0, 255, 0)
        elif gate.hand_confirm_streak > 0:
            status = (f"HAND CONFIRM {gate.hand_confirm_streak}/"
                      f"{gate.hand_confirm_required} - HOLDING")
            color = (0, 215, 255)
        else:
            if thumb_keys is not None and thumb_keys.manual_override:
                status = "NO HAND - MANUAL THUMB (space: stop)"
                color = (255, 220, 0)
            else:
                status = "NO HAND - HOLDING (space: stop)"
                color = (0, 0, 255)
        cv2.putText(view, status, (12, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    color, 2, cv2.LINE_AA)
    if thumb_keys is not None:
        y = 84 if gate is not None else 56
        q0_target = "--" if thumb_keys.q0.effective is None else str(thumb_keys.q0.effective)
        abd_target = "--" if thumb_keys.abd.effective is None else str(thumb_keys.abd.effective)
        roll_target = "--" if thumb_keys.roll.effective is None else str(thumb_keys.roll.effective)
        tip_target = "--" if thumb_keys.tip.effective is None else str(thumb_keys.tip.effective)
        if thumb_keys.return_to_open:
            mode = "RETURNING"
        elif thumb_keys.mediapipe_thumb_enabled:
            mode = "CAMERA"
        else:
            mode = "LOCKED/MANUAL"
        cv2.putText(view,
                    f"THUMB={mode} q0={q0_target} trim={thumb_keys.q0.offset:+d} (W+/S-)",
                    (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    (255, 220, 0), 2, cv2.LINE_AA)
        cv2.putText(view,
                    f"q5/side={abd_target} trim={thumb_keys.abd.offset:+d} (L+/J-)",
                    (12, y + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    (255, 220, 0), 2, cv2.LINE_AA)
        cv2.putText(view,
                    f"q10/roll={roll_target} trim={thumb_keys.roll.offset:+d} (I+/K-)",
                    (12, y + 56), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    (255, 220, 0), 2, cv2.LINE_AA)
        cv2.putText(view,
                    f"q15/tip={tip_target} trim={thumb_keys.tip.offset:+d} (D+/A-) Q:cam R:return",
                    (12, y + 84), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    (255, 220, 0), 2, cv2.LINE_AA)
    cv2.imshow(window, view)
    key = cv2.waitKey(1) & 0xFF
    if gate is not None and key == ord(" ") and gate.try_toggle():
        active = gate.active
        print(f"[hardware] {'START — recording + hand following camera' if active else 'STOP — recording off + hand reset'} "
              f"(press space to toggle)", flush=True)
    _handle_thumb_key(key, thumb_keys)
    return key != 27


@dataclass
class SourceBundle:
    source: object
    stream: Iterator[Tuple[str, np.ndarray, float]]
    cv2: object = None

    def close(self) -> None:
        try:
            self.source.close()
        finally:
            if self.cv2 is not None:
                self.cv2.destroyAllWindows()


def _rounded_points(points, *, digits: int = 4, indices: Optional[Tuple[int, ...]] = None):
    if points is None:
        return None
    arr = np.asarray(points, dtype=float)
    if arr.ndim != 2 or arr.shape[0] == 0:
        return None
    if indices is not None:
        if max(indices) >= arr.shape[0]:
            return None
        arr = arr[list(indices)]
    if not np.all(np.isfinite(arr)):
        return None
    return np.round(arr, digits).tolist()


def _print_mediapipe_log(source, pf) -> None:
    tip_idx = (4, 8, 12, 16, 20)
    payload = {
        "t": round(float(pf.t), 4),
        "side": pf.side,
        "score": round(float(pf.score), 4),
        "held": bool(getattr(pf, "held", False)),
        "tip_idx": list(tip_idx),
        "raw_world21": _rounded_points(getattr(source, "last_world_landmarks_raw", None)),
        "adj_world21": _rounded_points(getattr(source, "last_world_landmarks", None)),
        "hand_base21": _rounded_points(pf.landmarks),
        "raw_tips_px": _rounded_points(
            getattr(source, "last_landmarks_raw_px", None), digits=1, indices=tip_idx),
        "adj_tips_px": _rounded_points(
            getattr(source, "last_landmarks_px", None), digits=1, indices=tip_idx),
    }
    print("[mediapipe] " + json.dumps(payload, separators=(",", ":")), flush=True)


def make_processed_stream(args: argparse.Namespace, motion_gate=None,
                          thumb_keys=None, idle_callback=None) -> SourceBundle:
    from src.perception.pipeline import HandPipeline
    from src.perception.one_euro import OneEuroConfig

    cv2 = None
    if args.show_camera:
        import cv2 as _cv2
        cv2 = _cv2

    if args.source == "webcam":
        from src.perception.mediapipe_source import MediaPipeHandSource
        source = MediaPipeHandSource(
            camera_index=args.camera_index,
            min_detection_confidence=args.min_hand_detection_confidence,
            min_tracking_confidence=args.min_hand_tracking_confidence,
            fps=args.rate,
            fingertip_extend=args.fingertip_extend,
            fingertip_lateral=args.fingertip_lateral,
            fingertip_straighten=args.fingertip_straighten,
        )
    elif args.source == "realsense":
        from src.perception.realsense_source import RealSenseHandSource
        source = RealSenseHandSource(fps=int(round(args.rate)))
    else:
        raise ValueError(f"unsupported source {args.source!r}")

    pipeline = HandPipeline(
        source,
        smoothing=not args.no_smoothing,
        one_euro=OneEuroConfig(
            min_cutoff=args.one_euro_min_cutoff,
            beta=args.one_euro_beta,
            d_cutoff=args.one_euro_d_cutoff,
        ),
        image_mirrored=args.image_mirrored,
        force_side=args.side,
        min_score=args.min_hand_score,
    )
    last_mp_log = 0.0
    fresh_gate = FreshHandGate(args.hand_confirm_frames)

    def _stream() -> Iterator[Tuple[str, np.ndarray, float]]:
        nonlocal last_mp_log
        n = 0
        for det in source:
            pf = pipeline.process(det)
            was_tracking = fresh_gate.tracking
            safe_to_follow = fresh_gate.update(pf)
            if motion_gate is not None:
                motion_gate.hand_tracking = safe_to_follow
                motion_gate.hand_confirm_streak = fresh_gate.streak
                motion_gate.hand_confirm_required = fresh_gate.confirm_frames
            if not safe_to_follow:
                if was_tracking:
                    print("[hardware] HAND LOST — holding last published pose; "
                          "no stale landmark commands", flush=True)
                preview_pf = pf if FreshHandGate.is_fresh(pf) else None
                if not _camera_preview(cv2, source, "hardware camera", preview_pf,
                                       motion_gate, thumb_keys):
                    break
                # A missing hand intentionally produces no drive() frame. Keep
                # keyboard-only thumb motion and stopped-mode open reset alive
                # through a separate hardware tick instead.
                if idle_callback is not None:
                    idle_callback()
                continue
            if not was_tracking:
                print(f"[hardware] HAND CONFIRMED ({fresh_gate.confirm_frames} fresh frames) "
                      "— following enabled", flush=True)
            if not _camera_preview(cv2, source, "hardware camera", pf,
                                   motion_gate, thumb_keys):
                break
            if args.log_mediapipe_output:
                now = time.monotonic()
                if now - last_mp_log >= max(0.0, float(args.log_period)):
                    last_mp_log = now
                    _print_mediapipe_log(source, pf)
            yield pf.side, pf.landmarks, pf.t
            n += 1
            if args.max_frames is not None and n >= args.max_frames:
                break

    return SourceBundle(source=source, stream=_stream(), cv2=cv2)


class MotionGate:
    """Shared start/stop switch for publishing retargeted poses to the real hand.

    ``_camera_preview`` flips ``active`` when the user presses ``SPACE``; the
    hardware sink checks it before publishing so ``SPACE`` starts/stops the real
    hand following the camera without stopping perception or the preview.
    """

    def __init__(self, active: bool = False, on_toggle=None,
                 debounce_sec: float = 0.4, stop_mode: str = "open"):
        self.active = bool(active)
        self.on_toggle = on_toggle
        self.debounce_sec = float(debounce_sec)
        if stop_mode not in ("open", "hold"):
            raise ValueError("stop_mode must be 'open' or 'hold'")
        self.stop_mode = str(stop_mode)
        self._last_toggle = -1.0e9
        self.hand_tracking = False
        self.hand_confirm_streak = 0
        self.hand_confirm_required = 1

    def try_toggle(self) -> bool:
        """Flip state, ignoring key-repeat bounces within ``debounce_sec``.

        Returns True if the state actually flipped, False if the press was
        swallowed as a bounce. cv2.waitKey often reports one physical press
        across two consecutive frames; without this an odd press is a no-op.
        """
        now = time.monotonic()
        if now - self._last_toggle < self.debounce_sec:
            return False
        self._last_toggle = now
        self.active = not self.active
        if self.on_toggle is not None:
            self.on_toggle(self.active)
        return True


class FreshHandGate:
    """Only pass consecutive fresh detections to real-hardware teleop.

    ``HandPipeline`` intentionally holds its last good frame on a miss. That is
    useful to downstream perception consumers, but repeatedly sending a held
    frame lets the range step limiter keep walking the real hand toward an old
    target after the operator has left the camera. Hardware teleop therefore
    rejects held frames and waits for a short fresh-detection streak to resume.
    """

    def __init__(self, confirm_frames: int = 3):
        self.confirm_frames = max(1, int(confirm_frames))
        self.streak = 0
        self.tracking = False

    @staticmethod
    def is_fresh(frame) -> bool:
        return (
            frame is not None
            and bool(getattr(frame, "detected", False))
            and not bool(getattr(frame, "held", True))
        )

    def update(self, frame) -> bool:
        if not self.is_fresh(frame):
            self.streak = 0
            self.tracking = False
            return False
        self.streak += 1
        if self.streak >= self.confirm_frames:
            self.tracking = True
        return self.tracking


class Q0KeyTrim:
    """Runtime SDK-range trim for G20 q0 (thumb base)."""

    def __init__(self, step: int = 5, limit: int = 255,
                 label: str = "q0/thumb-base"):
        self.step = max(1, int(step))
        self.limit = max(0, int(limit))
        self.label = str(label)
        self.offset = 0
        self.base: Optional[int] = None
        self.effective: Optional[int] = None

    def _clamp_offset_for_base(self, offset: int) -> int:
        offset = max(-self.limit, min(self.limit, int(offset)))
        if self.base is None:
            return offset
        # Anti-windup: once base + offset reaches 0/255, do not accumulate
        # invisible trim beyond that boundary. One reverse key press therefore
        # changes q0 immediately instead of first consuming a large dead range.
        return max(-self.base, min(255 - self.base, offset))

    def adjust(self, direction: int) -> int:
        delta = self.step if int(direction) > 0 else -self.step
        self.offset = self._clamp_offset_for_base(self.offset + delta)
        if self.base is not None:
            self.effective = max(0, min(255, self.base + self.offset))
        target = "--" if self.effective is None else str(self.effective)
        print(f"[hardware] {self.label} live trim={self.offset:+d} ticks "
              f"target={target} "
              f"(+/- keyboard adjustment)", flush=True)
        return self.offset


class Q15KeyTrim(Q0KeyTrim):
    """Runtime SDK-range trim for G20 q15 (thumb fingertip)."""

    def __init__(self, step: int = 10, limit: int = 255):
        super().__init__(step=step, limit=limit, label="q15/thumb-tip")


class Q5KeyTrim(Q0KeyTrim):
    """Runtime SDK-range trim for G20 q5 (thumb abduction/side swing)."""

    def __init__(self, step: int = 10, limit: int = 255):
        super().__init__(step=step, limit=limit, label="q5/thumb-side")


class Q10KeyTrim(Q0KeyTrim):
    """Runtime SDK-range trim for G20 q10 (thumb roll/opposition)."""

    def __init__(self, step: int = 10, limit: int = 255):
        super().__init__(step=step, limit=limit, label="q10/thumb-roll")


class ThumbKeyboardControl:
    """Shared keyboard state for manual thumb trim and slow open return."""

    def __init__(self, q0_step: int = 5, tip_step: int = 10,
                 abd_step: int = 10, roll_step: int = 10):
        self.q0 = Q0KeyTrim(step=q0_step)
        self.abd = Q5KeyTrim(step=abd_step)
        self.roll = Q10KeyTrim(step=roll_step)
        self.tip = Q15KeyTrim(step=tip_step)
        self.return_to_open = False
        self.mediapipe_thumb_enabled = True
        # True means keyboard commands own the four thumb channels even when
        # MediaPipe has no fresh hand, or SPACE has stopped camera following.
        self.manual_override = False

    @property
    def trims(self):
        return (self.q0, self.abd, self.roll, self.tip)

    def seed_from_pose(self, pose: List[int], *, manual_override: bool = False) -> None:
        """Make the next key press relative to a known 20-D hardware pose."""
        for trim, idx in zip(self.trims, (0, 5, 10, 15)):
            value = max(0, min(255, int(pose[idx])))
            trim.offset = 0
            trim.base = value
            trim.effective = value
        self.return_to_open = False
        self.manual_override = bool(manual_override)

    def start_manual_control(self) -> None:
        self.return_to_open = False
        self.manual_override = True

    def toggle_mediapipe_thumb(self) -> bool:
        """Toggle camera ownership of q0/q5/q10/q15.

        On lock, rebase every trim at its last effective command so subsequent
        keyboard presses are absolute manual movements rather than offsets from
        a changing camera pose. On unlock, discard manual offsets and let the
        next fresh MediaPipe frame take ownership through the normal step cap.
        """
        self.return_to_open = False
        self.mediapipe_thumb_enabled = not self.mediapipe_thumb_enabled
        if not self.mediapipe_thumb_enabled:
            for trim in self.trims:
                if trim.effective is not None:
                    trim.base = int(trim.effective)
                trim.offset = 0
            self.manual_override = True
            print("[hardware] Q: MediaPipe thumb OFF — holding q0/q5/q10/q15; "
                  "keyboard thumb control remains active", flush=True)
        else:
            for trim in self.trims:
                trim.offset = 0
            self.manual_override = False
            print("[hardware] Q: MediaPipe thumb ON — q0/q5/q10/q15 follow "
                  "fresh hand frames again", flush=True)
        return self.mediapipe_thumb_enabled

    def start_return_to_open(self) -> None:
        for trim in self.trims:
            trim.offset = 0
        self.return_to_open = True
        self.manual_override = True
        print("[hardware] R: thumb returning slowly to initial/open pose "
              "(W/S/J/L/I/K/A/D resumes teleop)", flush=True)


def _apply_live_trim_at(pose: List[int], trim: Optional[Q0KeyTrim],
                        index: int) -> List[int]:
    out = list(pose)
    if trim is not None:
        trim.base = max(0, min(255, int(out[index])))
        trim.offset = trim._clamp_offset_for_base(trim.offset)
        out[index] = max(0, min(255, trim.base + trim.offset))
        trim.effective = out[index]
    return out


def apply_q0_live_trim(pose: List[int], trim: Optional[Q0KeyTrim]) -> List[int]:
    """Trim G20 q0 (first GUI joint / thumb base)."""
    return _apply_live_trim_at(pose, trim, 0)


def apply_q15_live_trim(pose: List[int], trim: Optional[Q15KeyTrim]) -> List[int]:
    """Trim G20 q15 (thumb fingertip)."""
    return _apply_live_trim_at(pose, trim, 15)


def apply_q5_live_trim(pose: List[int], trim: Optional[Q5KeyTrim]) -> List[int]:
    """Trim G20 q5 (thumb abduction/side swing)."""
    return _apply_live_trim_at(pose, trim, 5)


def apply_q10_live_trim(pose: List[int], trim: Optional[Q10KeyTrim]) -> List[int]:
    """Trim G20 q10 (thumb roll/opposition)."""
    return _apply_live_trim_at(pose, trim, 10)


def apply_thumb_keyboard_control(
    pose: List[int], control: Optional[ThumbKeyboardControl],
    open_range: List[int],
) -> List[int]:
    """Apply manual trims, or hold all four thumb channels at open targets."""
    if control is None:
        return list(pose)
    if control.return_to_open:
        out = list(pose)
        for idx in (0, 5, 10, 15):
            out[idx] = int(open_range[idx])
        control.q0.effective = out[0]
        control.abd.effective = out[5]
        control.roll.effective = out[10]
        control.tip.effective = out[15]
        return out
    out = apply_q0_live_trim(pose, control.q0)
    out = apply_q5_live_trim(out, control.abd)
    out = apply_q10_live_trim(out, control.roll)
    return apply_q15_live_trim(out, control.tip)


def manual_thumb_target(
    pose: List[int], control: Optional[ThumbKeyboardControl],
    open_range: List[int],
) -> List[int]:
    """Overlay persistent keyboard targets without rebasing them each tick.

    ``apply_thumb_keyboard_control`` intentionally rebases trims on each fresh
    MediaPipe pose. A no-hand loop must not do that: rebasing an offset against
    the previous output would add the same offset repeatedly and make the thumb
    drift. This helper holds each key-selected absolute target instead.
    """
    out = list(pose)
    if control is None:
        return out
    if control.return_to_open:
        targets = [int(open_range[i]) for i in (0, 5, 10, 15)]
    else:
        targets = []
        for trim, idx in zip(control.trims, (0, 5, 10, 15)):
            if trim.effective is None:
                value = max(0, min(255, int(out[idx])))
                trim.base = value
                trim.effective = value
            targets.append(int(trim.effective))
    for trim, idx, target in zip(control.trims, (0, 5, 10, 15), targets):
        out[idx] = max(0, min(255, int(target)))
        trim.effective = out[idx]
    return out


class CompositeSink:
    """Forward the same radian command to multiple drive-compatible sinks."""

    def __init__(self, *sinks):
        self.sinks = [s for s in sinks if s is not None]

    def set_joints(self, joint_rad) -> None:
        for sink in self.sinks:
            sink.set_joints(joint_rad)


class LinkerHandHardwareSink:
    """A ``drive``-compatible sink that publishes safe L20 commands to ROS2."""

    def __init__(self, node, *, side: str, enable_motion: bool,
                 max_range_step: int, log_period: float,
                 relative_mode: bool, calibration_frames: int,
                 relative_scale: float, max_relative_delta: int,
                 open_range: List[int], hardware_map: str,
                 roll_range_ticks: float, base_gain: float,
                 base_gains: Tuple[float, float, float, float],
                 spread_gain: float, tip_gain: float,
                 tip_gains: Tuple[float, float, float, float],
                 spread_signs: Tuple[float, float, float, float],
                 thumb_base_gain: float, thumb_abd_gain: float,
                 thumb_roll_gain: float, thumb_tip_gain: float,
                 thumb_base_offset: int, thumb_abd_offset: int,
                 thumb_roll_offset: int, thumb_tip_offset: int,
                 nonthumb_close_deadzone: int,
                 collision_guard: bool, thumb_safe_mode: str,
                 max_thumb_delta: int, max_thumb_abd_delta: int,
                 max_thumb_base_delta: Optional[int],
                 max_spread_delta: int,
                 spread_close_threshold: float, spread_recenter_gain: float,
                 min_spread_gap: int,
                 thumb_index_guard: bool,
                 thumb_index_threshold: float,
                 thumb_index_release: int,
                 log_sim_position: bool = False,
                 motion_gate: Optional["MotionGate"] = None,
                 thumb_keys: Optional["ThumbKeyboardControl"] = None):
        self.node = node
        self.side = side
        self.enable_motion = enable_motion
        self.motion_gate = motion_gate
        self.thumb_keys = thumb_keys
        self.max_range_step = int(max_range_step)
        self.log_period = float(log_period)
        self.relative_mode = bool(relative_mode)
        self.calibration_frames = max(1, int(calibration_frames))
        self.relative_scale = float(relative_scale)
        self.max_relative_delta = max(0, int(max_relative_delta))
        self.open_range = list(open_range)
        self.hardware_map = hardware_map
        self.roll_range_ticks = float(roll_range_ticks)
        self.base_gain = float(base_gain)
        self.base_gains = tuple(float(v) for v in base_gains)
        self.spread_gain = float(spread_gain)
        self.spread_signs = tuple(float(v) for v in spread_signs)
        self.tip_gain = float(tip_gain)
        self.tip_gains = tuple(float(v) for v in tip_gains)
        self.thumb_base_gain = float(thumb_base_gain)
        self.thumb_abd_gain = float(thumb_abd_gain)
        self.thumb_roll_gain = float(thumb_roll_gain)
        self.thumb_tip_gain = float(thumb_tip_gain)
        self.thumb_base_offset = int(thumb_base_offset)
        self.thumb_abd_offset = int(thumb_abd_offset)
        self.thumb_roll_offset = int(thumb_roll_offset)
        self.thumb_tip_offset = int(thumb_tip_offset)
        self.nonthumb_close_deadzone = int(nonthumb_close_deadzone)
        self.collision_guard = bool(collision_guard)
        self.thumb_safe_mode = thumb_safe_mode
        self.max_thumb_delta = int(max_thumb_delta)
        self.max_thumb_abd_delta = int(max_thumb_abd_delta)
        self.max_thumb_base_delta = None if max_thumb_base_delta is None else int(max_thumb_base_delta)
        self.max_spread_delta = int(max_spread_delta)
        self.spread_close_threshold = float(spread_close_threshold)
        self.spread_recenter_gain = float(spread_recenter_gain)
        self.min_spread_gap = int(min_spread_gap)
        self.thumb_index_guard = bool(thumb_index_guard)
        self.thumb_index_threshold = float(thumb_index_threshold)
        self.thumb_index_release = int(thumb_index_release)
        self.log_sim_position = bool(log_sim_position)
        self.prev_range: Optional[List[int]] = None
        self.baseline_samples: List[List[int]] = []
        self.baseline_range: Optional[List[int]] = None
        self.last_log = 0.0

    def _relative_pose(self, raw_pose: List[int]) -> Tuple[List[int], str]:
        if not self.relative_mode:
            return raw_pose, "absolute"

        if self.baseline_range is None:
            self.baseline_samples.append(raw_pose)
            if len(self.baseline_samples) >= self.calibration_frames:
                arr = np.asarray(self.baseline_samples, dtype=float)
                self.baseline_range = [int(round(v)) for v in arr.mean(axis=0)]
                print(f"[hardware] calibrated open baseline from "
                      f"{len(self.baseline_samples)} frames: {self.baseline_range}",
                      flush=True)
            return list(self.open_range), f"calibrating {len(self.baseline_samples)}/{self.calibration_frames}"

        out = list(self.open_range)
        for i in ACTIVE_IDX:
            delta = (raw_pose[i] - self.baseline_range[i]) * self.relative_scale
            if self.max_relative_delta > 0:
                delta = _clamp(delta, -self.max_relative_delta, self.max_relative_delta)
            out[i] = max(0, min(255, int(round(self.open_range[i] + delta))))
        for i in RESERVED_IDX:
            out[i] = self.open_range[i]
        return out, "relative"

    def set_joints(self, joint_rad) -> None:
        if self.motion_gate is not None:
            # Keep the recorder in sync every frame (cheap Bool at control rate).
            self.node.publish_session_active(self.motion_gate.active)
            if not self.motion_gate.active:
                if self.thumb_keys is not None and self.thumb_keys.manual_override:
                    self.publish_manual_thumb_tick()
                    return
                if self.motion_gate.stop_mode == "hold":
                    return
                # Stopped: ramp the real hand back to the open pose (手复位) and
                # hold there. prev_range advances so the next start ramps smoothly.
                pose = _limit_range_step(list(self.open_range), self.prev_range,
                                         self.max_range_step)
                self.prev_range = pose
                self.node.publish_pose(pose)
                return
        if self.hardware_map == "g20-sim":
            raw_pose = g20_sim_radians_to_sdk_range(
                joint_rad,
                open_range=self.open_range,
                roll_range_ticks=self.roll_range_ticks,
                base_gain=self.base_gain,
                base_gains=self.base_gains,
                spread_gain=self.spread_gain,
                spread_signs=self.spread_signs,
                tip_gain=self.tip_gain,
                tip_gains=self.tip_gains,
                thumb_base_gain=self.thumb_base_gain,
                thumb_abd_gain=self.thumb_abd_gain,
                thumb_roll_gain=self.thumb_roll_gain,
                thumb_tip_gain=self.thumb_tip_gain,
                thumb_base_offset=self.thumb_base_offset,
                thumb_abd_offset=self.thumb_abd_offset,
                thumb_roll_offset=self.thumb_roll_offset,
                thumb_tip_offset=self.thumb_tip_offset,
            )
        else:
            raw_pose = l20_radians_to_sdk_range(joint_rad, self.side, self.open_range)
        pose, mode = self._relative_pose(raw_pose)
        if (self.thumb_keys is not None
                and not self.thumb_keys.mediapipe_thumb_enabled):
            # Q-lock: ignore the four thumb channels predicted from the current
            # MediaPipe frame while the remaining joints keep following it.
            pose = manual_thumb_target(pose, self.thumb_keys, self.open_range)
        else:
            pose = apply_thumb_keyboard_control(pose, self.thumb_keys,
                                                self.open_range)
        pose = apply_nonthumb_close_deadzone(
            pose,
            self.open_range,
            deadzone=self.nonthumb_close_deadzone,
        )
        pose = apply_hardware_collision_guard(
            pose,
            self.open_range,
            enabled=self.collision_guard,
            thumb_safe_mode=self.thumb_safe_mode,
            max_thumb_delta=self.max_thumb_delta,
            max_thumb_abd_delta=self.max_thumb_abd_delta,
            max_thumb_base_delta=self.max_thumb_base_delta,
            max_spread_delta=self.max_spread_delta,
            spread_close_threshold=self.spread_close_threshold,
            spread_recenter_gain=self.spread_recenter_gain,
            min_spread_gap=self.min_spread_gap,
        )
        pose = apply_thumb_index_guard(
            pose,
            self.open_range,
            enabled=self.thumb_index_guard,
            threshold=self.thumb_index_threshold,
            release_ticks=self.thumb_index_release,
        )
        guarded_pose = list(pose)
        pose = _limit_range_step(guarded_pose, self.prev_range, self.max_range_step)
        self.prev_range = pose
        self.node.publish_pose(pose)
        now = time.monotonic()
        if now - self.last_log >= self.log_period:
            self.last_log = now
            state = self.node.last_state
            state_msg = ""
            if state is not None and len(state) >= 20:
                err = [abs(int(state[i]) - int(pose[i])) for i in ACTIVE_IDX]
                state_msg = (
                    f" state_err_max={max(err)}"
                    f" state_active={[state[i] for i in ACTIVE_IDX]}"
                )
            guard_clip = [
                f"{i}:{raw_pose[i]}->{guarded_pose[i]}"
                for i in ACTIVE_IDX
                if int(raw_pose[i]) != int(guarded_pose[i])
            ]
            step_clip = [
                f"{i}:{guarded_pose[i]}->{pose[i]}"
                for i in ACTIVE_IDX
                if int(guarded_pose[i]) != int(pose[i])
            ]
            clipped_msg = ""
            if guard_clip:
                clipped_msg += f" guard_clip={guard_clip}"
            if step_clip:
                clipped_msg += f" step_clip={step_clip}"
            sim_msg = ""
            if self.log_sim_position:
                q = np.asarray(joint_rad, dtype=float).reshape(-1)
                sim_active = [round(float(q[i]), 3) for i in ACTIVE_IDX]
                sim_msg = f" sim_rad_active={sim_active}"
            print(f"[hardware] {'PUBLISH' if self.enable_motion else 'dry-run'} "
                  f"{mode} {self.node.cmd_topic}: {pose} raw={raw_pose}"
                  f"{sim_msg}{clipped_msg}{state_msg}", flush=True)

    def publish_manual_thumb_tick(self) -> bool:
        """Advance a keyboard-only thumb target by one safe range step.

        This path is independent of MediaPipe, so it remains available with no
        detected hand and while SPACE has stopped camera following. Non-thumb
        joints stay at their last commanded values.
        """
        control = self.thumb_keys
        if control is None or not control.manual_override:
            return False
        base = self.prev_range
        if base is None:
            base = self.node.last_state
        if base is None or len(base) < 20:
            base = self.open_range
        target = manual_thumb_target(list(base), control, self.open_range)
        target = apply_hardware_collision_guard(
            target,
            self.open_range,
            enabled=self.collision_guard,
            thumb_safe_mode=self.thumb_safe_mode,
            max_thumb_delta=self.max_thumb_delta,
            max_thumb_abd_delta=self.max_thumb_abd_delta,
            max_thumb_base_delta=self.max_thumb_base_delta,
            max_spread_delta=self.max_spread_delta,
            spread_close_threshold=self.spread_close_threshold,
            spread_recenter_gain=self.spread_recenter_gain,
            min_spread_gap=self.min_spread_gap,
        )
        target = apply_thumb_index_guard(
            target,
            self.open_range,
            enabled=self.thumb_index_guard,
            threshold=self.thumb_index_threshold,
            release_ticks=self.thumb_index_release,
        )
        pose = _limit_range_step(target, self.prev_range, self.max_range_step)
        self.prev_range = pose
        self.node.publish_pose(pose)
        now = time.monotonic()
        if now - self.last_log >= self.log_period:
            self.last_log = now
            state = self.node.last_state
            state_msg = ""
            if state is not None and len(state) >= 20:
                state_msg = (
                    " state_thumb="
                    f"{[state[i] for i in (0, 5, 10, 15)]}"
                )
            print("[hardware] "
                  f"{'PUBLISH' if self.enable_motion else 'dry-run'} manual-thumb "
                  f"target={[target[i] for i in (0, 5, 10, 15)]} "
                  f"cmd={[pose[i] for i in (0, 5, 10, 15)]}"
                  f"{state_msg}", flush=True)
        return True

    def idle_tick(self) -> None:
        """Hardware work that must continue when perception yields no frame."""
        if self.publish_manual_thumb_tick():
            return
        if self.motion_gate is not None and not self.motion_gate.active:
            if self.motion_gate.stop_mode == "hold":
                return
            pose = _limit_range_step(list(self.open_range), self.prev_range,
                                     self.max_range_step)
            self.prev_range = pose
            self.node.publish_pose(pose)

    def release_open(self, frames: int = 20) -> None:
        for _ in range(max(0, int(frames))):
            pose = _limit_range_step(list(self.open_range), self.prev_range, self.max_range_step)
            self.prev_range = pose
            self.node.publish_pose(pose)
            time.sleep(0.02)

    def hold_open(self, seconds: float) -> None:
        deadline = time.monotonic() + max(0.0, float(seconds))
        while time.monotonic() < deadline:
            self.release_open(frames=1)

    def hold_open_until_thumb_ready(self, seconds: float,
                                    tolerance: int = 12,
                                    confirm_samples: int = 3) -> bool:
        """Ramp open until all four measured thumb joints are near G20 open.

        ``seconds`` is a maximum wait, not a blind delay. This exposes a stuck
        or unresponsive q0/q5/q10/q15 instead of silently starting teleop from
        the wrong pose.
        """
        deadline = time.monotonic() + max(0.0, float(seconds))
        tolerance = max(0, int(tolerance))
        required = max(1, int(confirm_samples))
        ready = 0
        last_values = None
        last_errors = None
        while time.monotonic() < deadline:
            self.release_open(frames=1)
            state = self.node.last_state
            if state is None or len(state) < 20:
                continue
            last_values = [int(state[i]) for i in (0, 5, 10, 15)]
            targets = [int(self.open_range[i]) for i in (0, 5, 10, 15)]
            last_errors = [abs(a - b) for a, b in zip(last_values, targets)]
            if max(last_errors) <= tolerance:
                ready += 1
                if ready >= required:
                    print("[hardware] startup thumb reset complete: "
                          f"state={last_values} target={targets}", flush=True)
                    return True
            else:
                ready = 0
        targets = [int(self.open_range[i]) for i in (0, 5, 10, 15)]
        print("[hardware] WARNING: startup thumb reset timed out; "
              f"state={last_values} target={targets} error={last_errors}. "
              "Keyboard control remains available; check the SDK/CAN motor if "
              "q5 or q10 never changes.", flush=True)
        return False

    def publish_range_pose(self, pose: List[int], seconds: float) -> None:
        deadline = time.monotonic() + max(0.0, float(seconds))
        pose = [max(0, min(255, int(round(v)))) for v in pose[:20]]
        while time.monotonic() < deadline:
            stepped = _limit_range_step(pose, self.prev_range, self.max_range_step)
            self.prev_range = stepped
            self.node.publish_pose(stepped)
            time.sleep(0.02)


class L20RosNode:
    """Thin wrapper around rclpy so imports stay lazy until hardware CLI runs."""

    def __init__(self, args: argparse.Namespace):
        import rclpy
        from rclpy.node import Node
        from sensor_msgs.msg import JointState
        from std_msgs.msg import Bool, Empty, String

        self.rclpy = rclpy
        self.JointState = JointState
        self.String = String
        self.Bool = Bool
        self.Empty = Empty

        class _Node(Node):
            pass

        self.node = _Node("l20_camera_retarget_bridge")
        self.args = args
        self.cmd_topic = f"/cb_{args.side}_hand_control_cmd"
        self.state_topic = f"/cb_{args.side}_hand_state"
        self.setting_topic = "/cb_hand_setting_cmd"
        self.session_topic = f"/cb_{args.side}_recording_active"
        self.session_delete_topic = f"/cb_{args.side}_recording_delete_last"
        self.cmd_pub = self.node.create_publisher(JointState, self.cmd_topic, 10)
        self.setting_pub = self.node.create_publisher(String, self.setting_topic, 10)
        self.session_pub = self.node.create_publisher(Bool, self.session_topic, 10)
        self.session_delete_pub = self.node.create_publisher(
            Empty, self.session_delete_topic, 10
        )
        self._last_session_active: Optional[bool] = None
        self.last_state: Optional[List[int]] = None
        self.state_sub = self.node.create_subscription(
            JointState, self.state_topic, self._state_cb, 10)

    def _state_cb(self, msg) -> None:
        if len(msg.position) >= 20:
            self.last_state = [max(0, min(255, int(round(v)))) for v in msg.position[:20]]

    def publish_settings(self, *, thumb_current_limit: Optional[int] = None) -> None:
        torque = [int(self.args.current_limit)] * 5
        if thumb_current_limit is not None:
            torque[0] = int(thumb_current_limit)
        settings = (
            ("set_max_torque_limits", "torque", torque),
            ("set_speed", "speed", [int(self.args.speed_limit)] * 5),
        )
        for setting_cmd, key, values in settings:
            msg = self.String()
            msg.data = json.dumps({
                "setting_cmd": setting_cmd,
                "params": {"hand_type": self.args.side, key: values},
            })
            if self.args.enable_motion:
                self.setting_pub.publish(msg)
            print(f"[hardware] {'PUBLISH' if self.args.enable_motion else 'dry-run'} "
                  f"{self.setting_topic}: {msg.data}", flush=True)

    def publish_clear_faults(self) -> None:
        """Request a fault clear without publishing a position command."""
        msg = self.String()
        msg.data = json.dumps({
            "setting_cmd": "clear_faults",
            "params": {"hand_type": self.args.side},
        })
        if self.args.enable_motion:
            self.setting_pub.publish(msg)
        print(
            f"[hardware] {'PUBLISH' if self.args.enable_motion else 'dry-run'} "
            f"{self.setting_topic}: {msg.data}",
            flush=True,
        )

    def publish_session_active(self, active: bool) -> None:
        """Broadcast the record/follow session state so the recorder can sync."""
        msg = self.Bool()
        msg.data = bool(active)
        self.session_pub.publish(msg)
        if self._last_session_active != bool(active):
            self._last_session_active = bool(active)
            print(f"[hardware] session {'ACTIVE — recording + following' if active else 'IDLE — stopped + reset'} "
                  f"-> {self.session_topic}", flush=True)

    def publish_delete_last_episode(self) -> None:
        """Ask an attached recorder to delete its latest completed episode."""
        self.session_delete_pub.publish(self.Empty())
        print(
            f"[hardware] request delete latest completed episode -> "
            f"{self.session_delete_topic}",
            flush=True,
        )

    def wait_ready(self) -> bool:
        deadline = time.monotonic() + self.args.command_timeout
        while time.monotonic() < deadline:
            self.rclpy.spin_once(self.node, timeout_sec=0.05)
            cmd_ready = self.cmd_pub.get_subscription_count() > 0
            setting_ready = self.setting_pub.get_subscription_count() > 0
            if cmd_ready and setting_ready:
                break
            if not cmd_ready:
                print(f"[hardware] waiting for SDK subscriber on {self.cmd_topic}", flush=True)
            if not setting_ready:
                print(f"[hardware] waiting for SDK subscriber on {self.setting_topic}", flush=True)
        if self.cmd_pub.get_subscription_count() <= 0:
            print(f"[hardware] no SDK subscriber on {self.cmd_topic}", flush=True)
            return False
        if self.setting_pub.get_subscription_count() <= 0:
            print(f"[hardware] no SDK subscriber on {self.setting_topic}", flush=True)
            return False

        if not self.args.require_state:
            return True
        deadline = time.monotonic() + self.args.state_timeout
        while time.monotonic() < deadline:
            self.rclpy.spin_once(self.node, timeout_sec=0.05)
            if self.last_state is not None:
                print(f"[hardware] got state on {self.state_topic}: {self.last_state}", flush=True)
                return True
            print(f"[hardware] waiting for state on {self.state_topic}", flush=True)
        print(f"[hardware] no state received on {self.state_topic}", flush=True)
        return False

    def publish_pose(self, pose: List[int]) -> None:
        if not self.rclpy.ok():
            return
        self.rclpy.spin_once(self.node, timeout_sec=0.0)
        msg = self.JointState()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.name = L20_NAMES
        msg.position = [float(v) for v in pose]
        msg.velocity = [0.0] * 20
        msg.effort = [0.0] * 20
        if self.args.enable_motion:
            self.cmd_pub.publish(msg)

    def close(self) -> None:
        self.node.destroy_node()


def _require_hardware_token(enable_motion: bool) -> None:
    if enable_motion and not os.environ.get("HW_ENABLE_TOKEN"):
        raise RuntimeError(
            "Refusing to actuate: set HW_ENABLE_TOKEN manually, then pass --enable-motion."
        )


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", choices=("webcam", "realsense"), default="webcam")
    ap.add_argument("--camera-index", type=int, default=0)
    ap.add_argument("--min-hand-detection-confidence", type=float, default=0.70,
                    help="MediaPipe palm-detection confidence threshold; higher values "
                         "reduce background false positives")
    ap.add_argument("--min-hand-tracking-confidence", type=float, default=0.70,
                    help="MediaPipe landmark-tracking confidence threshold")
    ap.add_argument("--min-hand-score", type=float, default=0.50,
                    help="minimum MediaPipe handedness score accepted as a fresh hand")
    ap.add_argument("--hand-confirm-frames", type=int, default=3,
                    help="consecutive fresh hand frames required before hardware commands "
                         "resume after startup or hand loss")
    ap.add_argument("--side", choices=("right", "left"), default="right")
    ap.add_argument("--sdk-hand-joint", choices=("g20", "l20"), default="g20",
                    help="SDK driver command/open-pose convention; g20 matches linker_hand_g20_palm_touch")
    ap.add_argument("--hardware-map", choices=("auto", "g20-sim", "l20-sdk"), default="auto",
                    help="radian->range mapping; auto uses g20-sim for G20 and l20-sdk for L20")
    ap.add_argument("--roll-range-ticks", type=float, default=100.0,
                    help="G20 sim map: range ticks for non-thumb spread joints")
    ap.add_argument("--hardware-base-gain", type=float, default=1.8,
                    help="G20 sim map: multiply base flexion before range mapping")
    ap.add_argument("--hardware-base-gains", type=lambda s: _parse_four_floats(s, name="--hardware-base-gains"),
                    default=(1.0, 1.0, 1.0, 1.0),
                    help="G20 sim map: per-finger base multipliers for index,middle,ring,little after --hardware-base-gain")
    ap.add_argument("--hardware-spread-gain", type=float, default=1.0,
                    help="G20 sim map: multiply thumb/spread joints before range mapping")
    ap.add_argument("--hardware-spread-signs", type=lambda s: _parse_four_floats(s, name="--hardware-spread-signs"),
                    default=(-1.0, -1.0, -1.0, -1.0),
                    help="G20 sim map: signs for index/middle/ring/little spread q6-q9")
    ap.add_argument("--hardware-tip-gain", type=float, default=1.0,
                    help="G20 sim map: multiply coupled tip flexion before range mapping")
    ap.add_argument("--hardware-tip-gains", type=lambda s: _parse_four_floats(s, name="--hardware-tip-gains"),
                    default=(1.0, 1.0, 1.0, 1.0),
                    help="G20 sim map: per-finger tip multipliers for index,middle,ring,little after --hardware-tip-gain")
    ap.add_argument("--hardware-thumb-base-gain", type=float, default=1.0,
                    help="G20 sim map: thumb base flexion gain")
    ap.add_argument("--hardware-thumb-abd-gain", type=float, default=1.0,
                    help="G20 sim map: thumb abduction/opposition gain")
    ap.add_argument("--hardware-thumb-roll-gain", type=float, default=1.0,
                    help="G20 sim map: thumb roll/opposition gain")
    ap.add_argument("--hardware-thumb-tip-gain", type=float, default=1.0,
                    help="G20 sim map: thumb tip flexion gain")
    ap.add_argument("--hardware-thumb-base-offset", type=int, default=0,
                    help="G20 sim map: thumb base range offset after mapping")
    ap.add_argument("--hardware-thumb-abd-offset", type=int, default=0,
                    help="G20 sim map: thumb abduction range offset after mapping")
    ap.add_argument("--hardware-thumb-roll-offset", type=int, default=0,
                    help="G20 sim map: thumb roll range offset after mapping")
    ap.add_argument("--hardware-thumb-tip-offset", type=int, default=0,
                    help="G20 sim map: thumb tip range offset after mapping")
    ap.add_argument("--nonthumb-close-deadzone", type=int, default=0,
                    help="hardware output: ignore this many ticks of accidental non-thumb base/tip closure")
    ap.add_argument("--no-collision-guard", dest="collision_guard", action="store_false",
                    help="disable conservative range-space thumb/finger guard")
    ap.add_argument("--thumb-safe-mode", choices=("open", "limited", "free"), default="open",
                    help="hardware guard: keep thumb rotation open, limited, or free")
    ap.add_argument("--max-thumb-delta", type=int, default=25,
                    help="hardware guard limited mode: max thumb rotation ticks from open")
    ap.add_argument("--max-thumb-abd-delta", type=int, default=None,
                    help="hardware guard limited mode: max thumb side-swing ticks from open; default follows --max-thumb-delta")
    ap.add_argument("--max-thumb-base-delta", type=int, default=None,
                    help="hardware guard limited mode: max thumb base/downward ticks from open; unset leaves q0 uncapped")
    ap.add_argument("--max-spread-delta", type=int, default=70,
                    help="hardware guard: max four-finger spread ticks from open")
    ap.add_argument("--spread-close-threshold", type=float, default=0.75,
                    help="hardware guard: recenter spread when fingers are this closed")
    ap.add_argument("--spread-recenter-gain", type=float, default=0.20,
                    help="hardware guard: how strongly flexion recenters spread")
    ap.add_argument("--min-spread-gap", type=int, default=2,
                    help="hardware guard: preserve ordered gaps between spread channels")
    ap.add_argument("--thumb-index-guard", action="store_true", default=False,
                    help="hardware guard: open index slightly when thumb/index collision risk is high")
    ap.add_argument("--thumb-index-threshold", type=float, default=0.28,
                    help="thumb-index guard risk threshold, 0..1")
    ap.add_argument("--thumb-index-release", type=int, default=45,
                    help="thumb-index guard: max index opening ticks at full risk")
    ap.set_defaults(collision_guard=True)
    ap.add_argument("--image-mirrored", action="store_true")
    ap.add_argument("--show-camera", action="store_true")
    ap.add_argument("--q0-key-step", type=int, default=5,
                    help="SDK ticks added/subtracted from G20 q0/thumb base "
                         "per W/S key press in the camera window")
    ap.add_argument("--thumb-abd-key-step", type=int, default=10,
                    help="SDK ticks added/subtracted from G20 q5/thumb side swing "
                         "per L/J key press in the camera window")
    ap.add_argument("--thumb-roll-key-step", type=int, default=10,
                    help="SDK ticks added/subtracted from G20 q10/thumb roll/opposition "
                         "per I/K key press in the camera window")
    ap.add_argument("--thumb-tip-key-step", type=int, default=10,
                    help="SDK ticks added/subtracted from G20 q15/thumb tip "
                         "per D/A key press in the camera window")
    ap.add_argument("--motion-key-toggle", action="store_true",
                    help="start stopped; press SPACE in the camera window to start/stop the "
                         "hand following the camera. On stop the hand resets to open. Also "
                         "broadcasts /cb_<side>_recording_active for the recorder to follow "
                         "(needs --show-camera)")
    ap.add_argument("--motion-stop-mode", choices=("open", "hold"), default="open",
                    help="with --motion-key-toggle: reset open or hold position while stopped")
    ap.add_argument("--show-sim", action="store_true",
                    help="also show a PyBullet sim mirror driven by the same radian command")
    ap.add_argument("--debug-match", action="store_true",
                    help="print MediaPipe-vs-sim per-finger angle diagnostics")
    ap.add_argument("--debug-match-period", type=int, default=15,
                    help="print one match diagnostic every N processed frames")
    ap.add_argument("--fingertip-extend", default="0.0",
                    help=("webcam: extend MediaPipe fingertips along distal bones; "
                          "scalar or thumb,index,middle,ring,little"))
    ap.add_argument("--fingertip-lateral", default="0.0",
                    help=("webcam: shift MediaPipe fingertips sideways; "
                          "scalar or thumb,index,middle,ring,little, + toward index side"))
    ap.add_argument("--fingertip-straighten", default="0.0",
                    help=("webcam: blend fingertips toward PIP->DIP continuation; "
                          "scalar or thumb,index,middle,ring,little"))
    ap.add_argument("--thumb-gain", type=float, default=1.0)
    ap.add_argument("--thumb-cross-gain", type=float, default=0.0,
                    help="same sim thumb-cross assist as src.viz.app")
    ap.add_argument("--thumb-assist-smooth", type=float, default=0.0,
                    help="same sim thumb cross/contact assist smoothing as src.viz.app")
    ap.add_argument("--thumb-grasp-gain", type=float, default=0.0,
                    help="same sim thumb grasp assist as src.viz.app")
    ap.add_argument("--thumb-base-assist-gain", type=float, default=0.0,
                    help="same sim thumb q0 reach assist as src.viz.app")
    ap.add_argument("--thumb-tip-gain", type=float, default=1.0,
                    help="same sim thumb tip curl scale as src.viz.app")
    ap.add_argument("--thumb-orient-gain", type=float, default=0.0,
                    help="same sim thumb orientation assist as src.viz.app")
    ap.add_argument("--hardware-landmark-thumb", action="store_true", default=False,
                    help="enhance thumb q0/q5/q15 from front thumb landmarks 2/3/4")
    ap.add_argument("--no-hardware-landmark-thumb", dest="hardware_landmark_thumb",
                    action="store_false")
    ap.add_argument("--landmark-thumb-gain", type=float, default=1.0,
                    help="blend for landmark-derived thumb arch/opposition")
    ap.add_argument("--landmark-thumb-reach-gain", type=float, default=1.0,
                    help="scale thumb reach along index->little MCP line; 2 maps middle toward ring")
    ap.add_argument("--hardware-landmark-spread", action="store_true", default=False,
                    help="override non-thumb spread q6-q9 from landmark finger directions")
    ap.add_argument("--no-hardware-landmark-spread", dest="hardware_landmark_spread",
                    action="store_false")
    ap.add_argument("--landmark-spread-gain", type=float, default=2.5,
                    help="gain for landmark-derived non-thumb spread radians")
    ap.add_argument("--landmark-spread-limit", type=float, default=0.17,
                    help="absolute radian limit for landmark-derived q6-q9 spread")
    ap.add_argument("--landmark-spread-calibration-frames", type=int, default=30,
                    help="open-hand frames used as zero baseline for landmark spread")
    ap.add_argument("--rate", type=float, default=30.0)
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--no-filter", action="store_true")
    ap.add_argument("--no-smoothing", action="store_true",
                    help="disable perception-side one-euro landmark smoothing")
    ap.add_argument("--one-euro-min-cutoff", type=float, default=1.5,
                    help="perception One Euro min cutoff in Hz; lower is smoother but laggier")
    ap.add_argument("--one-euro-beta", type=float, default=0.05,
                    help="perception One Euro speed coefficient; higher reduces lag while moving")
    ap.add_argument("--one-euro-d-cutoff", type=float, default=1.0,
                    help="perception One Euro derivative cutoff in Hz")
    ap.add_argument("--absolute", action="store_true",
                    help="send absolute retargeted SDK ranges; default is calibrated relative mode")
    ap.add_argument("--calibration-frames", type=int, default=30,
                    help="relative mode: hold your real hand open for this many detected frames")
    ap.add_argument("--relative-scale", type=float, default=1.0,
                    help="relative mode: multiply raw range deltas after open-hand calibration")
    ap.add_argument("--max-relative-delta", type=int, default=120,
                    help="relative mode: max SDK range-unit delta from OPEN_RANGE; 0 disables")

    ap.add_argument("--enable-motion", action="store_true",
                    help="actually publish to the hardware command topic")
    ap.add_argument("--open-only", action="store_true",
                    help="publish the SDK open pose for --open-seconds, then exit")
    ap.add_argument("--open-seconds", type=float, default=2.0,
                    help="--open-only duration in seconds")
    ap.add_argument("--spread-test-only", action="store_true",
                    help="publish open and 6-9 spread-only test poses, then exit")
    ap.add_argument("--spread-test-seconds", type=float, default=1.5,
                    help="duration for each --spread-test-only pose")
    ap.add_argument("--open-on-start-seconds", type=float, default=4.0,
                    help=("hardware teleop: maximum wait for measured q0/q5/q10/q15 "
                          "to reach the open pose before camera calibration"))
    ap.add_argument("--require-state", action="store_true", default=True,
                    help="require /cb_<side>_hand_state before motion")
    ap.add_argument("--no-require-state", dest="require_state", action="store_false")
    ap.add_argument("--command-timeout", type=float, default=5.0)
    ap.add_argument("--state-timeout", type=float, default=5.0)
    ap.add_argument("--max-range-step", type=int, default=12,
                    help="per-frame SDK range-unit step cap; 0 disables")
    ap.add_argument("--current-limit", type=int, default=60,
                    help="conservative SDK current setting sent via /cb_hand_setting_cmd")
    ap.add_argument("--speed-limit", type=int, default=80,
                    help="conservative SDK speed setting sent via /cb_hand_setting_cmd")
    ap.add_argument("--log-period", type=float, default=0.5)
    ap.add_argument("--log-sim-position", action="store_true",
                    help="include the active 16-D sim joint radians in each hardware log line")
    ap.add_argument("--log-mediapipe-output", action="store_true",
                    help=("print raw/adjusted MediaPipe landmarks and pipeline hand_base "
                          "landmarks for sim-vs-hardware tuning"))
    ap.add_argument("--no-open-on-exit", dest="open_on_exit", action="store_false",
                    help="do not send a short open-hand release on exit")
    ap.set_defaults(open_on_exit=True)
    return ap


def main(argv: Optional[list] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        _require_hardware_token(args.enable_motion)
    except RuntimeError as exc:
        print(f"[hardware] {exc}", file=sys.stderr, flush=True)
        return 2

    import rclpy
    rclpy.init(args=None)
    ros = L20RosNode(args)
    bundle = None
    sink = None
    sim_model = None
    open_range = OPEN_RANGES[args.sdk_hand_joint]
    hardware_map = args.hardware_map
    if hardware_map == "auto":
        hardware_map = "g20-sim" if args.sdk_hand_joint == "g20" else "l20-sdk"
    try:
        if args.enable_motion and not ros.wait_ready():
            return 2
        elif not args.enable_motion:
            print("[hardware] dry-run: add --enable-motion and HW_ENABLE_TOKEN to move.", flush=True)
        print(f"[hardware] map={hardware_map} sdk_hand_joint={args.sdk_hand_joint} "
              f"base_gain={args.hardware_base_gain:.2f} "
              f"base_gains={tuple(args.hardware_base_gains)} "
              f"spread_gain={args.hardware_spread_gain:.2f} "
              f"spread_signs={tuple(args.hardware_spread_signs)} "
              f"tip_gain={args.hardware_tip_gain:.2f} "
              f"tip_gains={tuple(args.hardware_tip_gains)} "
              f"thumb_gains=({args.hardware_thumb_base_gain:.2f},"
              f"{args.hardware_thumb_abd_gain:.2f},"
              f"{args.hardware_thumb_roll_gain:.2f},"
              f"{args.hardware_thumb_tip_gain:.2f}) "
              f"roll_range_ticks={args.roll_range_ticks:.1f} "
              f"guard={args.collision_guard} thumb_mode={args.thumb_safe_mode} "
              f"thumb_delta={args.max_thumb_delta} "
              f"thumb_abd_delta={args.max_thumb_abd_delta if args.max_thumb_abd_delta is not None else args.max_thumb_delta} "
              f"thumb_base_delta={args.max_thumb_base_delta} "
              f"nonthumb_deadzone={args.nonthumb_close_deadzone} "
              f"spread_delta={args.max_spread_delta} "
              f"spread_recenter=({args.spread_close_threshold:.2f},"
              f"{args.spread_recenter_gain:.2f}) gap={args.min_spread_gap} "
              f"thumb_index_guard={args.thumb_index_guard} "
              f"thumb_index=({args.thumb_index_threshold:.2f},"
              f"{args.thumb_index_release}) "
              f"fingertip_extend={args.fingertip_extend} "
              f"fingertip_lateral={args.fingertip_lateral} "
              f"fingertip_straighten={args.fingertip_straighten} "
              f"sim_thumb=({args.thumb_gain:.2f},"
              f"{args.thumb_cross_gain:.2f},"
              f"{args.thumb_assist_smooth:.2f},"
              f"{args.thumb_grasp_gain:.2f},"
              f"{args.thumb_base_assist_gain:.2f},"
              f"{args.thumb_tip_gain:.2f},"
              f"{args.thumb_orient_gain:.2f}) "
              f"landmark_thumb={args.hardware_landmark_thumb} "
              f"landmark_thumb_gain={args.landmark_thumb_gain:.2f} "
              f"landmark_thumb_reach_gain={args.landmark_thumb_reach_gain:.2f} "
              f"landmark_spread={args.hardware_landmark_spread} "
              f"landmark_spread_gain={args.landmark_spread_gain:.2f} "
              f"landmark_spread_calib={args.landmark_spread_calibration_frames}",
              flush=True)

        motion_gate = None
        if args.motion_key_toggle:
            if not args.show_camera:
                print("[hardware] --motion-key-toggle needs --show-camera to read the "
                      "space key; ignoring toggle and running normally.", flush=True)
            else:
                motion_gate = MotionGate(
                    active=False,
                    on_toggle=ros.publish_session_active,
                    stop_mode=args.motion_stop_mode,
                )
                ros.publish_session_active(False)  # announce initial idle state
                print(f"[hardware] space toggle ON: starting STOPPED "
                      f"(stop_mode={args.motion_stop_mode}). "
                      "Focus the camera window and press SPACE to start/stop recording + motion. "
                      f"session state on {ros.session_topic}", flush=True)

        thumb_keys = ThumbKeyboardControl(
            q0_step=args.q0_key_step,
            abd_step=args.thumb_abd_key_step,
            roll_step=args.thumb_roll_key_step,
            tip_step=args.thumb_tip_key_step,
        )
        if args.show_camera:
            print(f"[hardware] all-thumb keyboard trim ON: "
                  f"W+/S- q0/base ({thumb_keys.q0.step} ticks); "
                  f"L+/J- q5/side ({thumb_keys.abd.step}); "
                  f"I+/K- q10/roll ({thumb_keys.roll.step}); "
                  f"D+/A- q15/tip ({thumb_keys.tip.step}); "
                  "Q toggles MediaPipe thumb OFF/ON; "
                  "R slowly returns all thumb joints to initial/open pose; "
                  "manual thumb keys work with NO HAND and while STOPPED; Esc quits",
                  flush=True)

        sink = LinkerHandHardwareSink(
            ros, side=args.side, enable_motion=args.enable_motion,
            max_range_step=args.max_range_step, log_period=args.log_period,
            relative_mode=not args.absolute,
            calibration_frames=args.calibration_frames,
            relative_scale=args.relative_scale,
            max_relative_delta=args.max_relative_delta,
            open_range=open_range,
            hardware_map=hardware_map,
            roll_range_ticks=args.roll_range_ticks,
            base_gain=args.hardware_base_gain,
            base_gains=args.hardware_base_gains,
            spread_gain=args.hardware_spread_gain,
            tip_gain=args.hardware_tip_gain,
            tip_gains=args.hardware_tip_gains,
            spread_signs=args.hardware_spread_signs,
            thumb_base_gain=args.hardware_thumb_base_gain,
            thumb_abd_gain=args.hardware_thumb_abd_gain,
            thumb_roll_gain=args.hardware_thumb_roll_gain,
            thumb_tip_gain=args.hardware_thumb_tip_gain,
            thumb_base_offset=args.hardware_thumb_base_offset,
            thumb_abd_offset=args.hardware_thumb_abd_offset,
            thumb_roll_offset=args.hardware_thumb_roll_offset,
            thumb_tip_offset=args.hardware_thumb_tip_offset,
            nonthumb_close_deadzone=args.nonthumb_close_deadzone,
            collision_guard=args.collision_guard,
            thumb_safe_mode=args.thumb_safe_mode,
            max_thumb_delta=args.max_thumb_delta,
            max_thumb_abd_delta=args.max_thumb_abd_delta if args.max_thumb_abd_delta is not None else args.max_thumb_delta,
            max_thumb_base_delta=args.max_thumb_base_delta,
            max_spread_delta=args.max_spread_delta,
            spread_close_threshold=args.spread_close_threshold,
            spread_recenter_gain=args.spread_recenter_gain,
            min_spread_gap=args.min_spread_gap,
            thumb_index_guard=args.thumb_index_guard,
            thumb_index_threshold=args.thumb_index_threshold,
            thumb_index_release=args.thumb_index_release,
            log_sim_position=args.log_sim_position,
            motion_gate=motion_gate,
            thumb_keys=thumb_keys,
        )
        if motion_gate is not None:
            def _handle_motion_toggle(active: bool) -> None:
                ros.publish_session_active(active)
                if (not active and args.enable_motion
                        and args.motion_stop_mode == "open"):
                    # A preview key event can arrive while MediaPipe has no hand
                    # detection. In that case drive() receives no frame and the
                    # normal inactive-gate path cannot reset the hardware. Ramp
                    # all the way to OPEN_RANGE directly from the toggle callback.
                    thumb_keys.seed_from_pose(
                        open_range,
                        manual_override=not thumb_keys.mediapipe_thumb_enabled,
                    )
                    step = max(1, sink.max_range_step)
                    reset_frames = int(np.ceil(255.0 / step)) + 1
                    sink.release_open(frames=reset_frames)

            motion_gate.on_toggle = _handle_motion_toggle
        if ros.last_state is not None:
            sink.prev_range = ros.last_state.copy()
        thumb_keys.seed_from_pose(sink.prev_range or open_range,
                                  manual_override=False)

        ros.publish_settings()
        if args.open_only:
            print(f"[hardware] {'PUBLISH' if args.enable_motion else 'dry-run'} "
                  f"open pose for {args.open_seconds:.2f}s", flush=True)
            sink.hold_open(args.open_seconds)
            return 0

        if args.spread_test_only:
            poses = []
            open_pose = list(open_range)
            poses.append(("open", open_pose))
            spread_open = list(open_range)
            spread_open[6], spread_open[7], spread_open[8], spread_open[9] = 255, 255, 0, 0
            poses.append(("spread_extreme_a", spread_open))
            spread_other = list(open_range)
            spread_other[6], spread_other[7], spread_other[8], spread_other[9] = 0, 0, 255, 255
            poses.append(("spread_extreme_b", spread_other))
            poses.append(("open", open_pose))
            for name, pose in poses:
                print(f"[hardware] {'PUBLISH' if args.enable_motion else 'dry-run'} "
                      f"{name} spread-only pose: {[pose[i] for i in NONTHUMB_SPREAD_IDX]}",
                      flush=True)
                sink.publish_range_pose(pose, args.spread_test_seconds)
            return 0

        if args.enable_motion and args.open_on_start_seconds > 0:
            print(f"[hardware] PUBLISH open pose before camera; waiting up to "
                  f"{args.open_on_start_seconds:.2f}s for thumb state", flush=True)
            sink.hold_open_until_thumb_ready(args.open_on_start_seconds)
            # Manual keys must start from the requested open pose, not from a
            # stale MediaPipe target left over before SPACE was stopped.
            thumb_keys.seed_from_pose(open_range, manual_override=False)

        drive_sink = sink
        if args.show_sim:
            from src.viz.render import L20VizModel
            sim_model = L20VizModel(args.side, gui=True)
            drive_sink = CompositeSink(sim_model, sink)

        bundle = make_processed_stream(args, motion_gate=motion_gate,
                                       thumb_keys=thumb_keys,
                                       idle_callback=sink.idle_tick)
        from src.viz.app import (
            _compose_adjust,
            _thumb_adjuster,
            _thumb_grasp_adjuster,
            _thumb_base_assist_adjuster,
            _thumb_tip_adjuster,
            _thumb_orient_adjuster,
        )
        candidate_adjust = _compose_adjust(
            _thumb_adjuster(args.thumb_gain, args.thumb_cross_gain,
                            args.thumb_assist_smooth),
            _thumb_grasp_adjuster(args.thumb_grasp_gain),
            _thumb_base_assist_adjuster(args.thumb_base_assist_gain,
                                        args.thumb_assist_smooth),
            _thumb_orient_adjuster(args.thumb_orient_gain),
            _thumb_tip_adjuster(args.thumb_tip_gain),
            _hardware_candidate_adjuster(
                1.0,
                landmark_thumb=args.hardware_landmark_thumb,
                landmark_thumb_gain=args.landmark_thumb_gain,
                landmark_thumb_reach_gain=args.landmark_thumb_reach_gain,
                landmark_spread=args.hardware_landmark_spread,
                landmark_spread_gain=args.landmark_spread_gain,
                landmark_spread_limit=args.landmark_spread_limit,
                landmark_spread_calibration_frames=args.landmark_spread_calibration_frames,
            ),
        )
        from src.viz.match_debug import make_match_debugger
        debug_callback = make_match_debugger(
            args.debug_match,
            period=args.debug_match_period,
        )

        drive(
            drive_sink,
            bundle.stream,
            use_filter=not args.no_filter,
            dt=1.0 / args.rate if args.rate > 0 else DEFAULT_DT,
            candidate_adjust=candidate_adjust,
            debug_callback=debug_callback,
        )
        return 0
    except KeyboardInterrupt:
        try:
            print("[hardware] interrupted; shutting down.", flush=True)
        except KeyboardInterrupt:
            pass
        return 130
    finally:
        if args.enable_motion and args.open_on_exit and sink is not None:
            print("[hardware] sending open release on exit.", flush=True)
            sink.release_open()
        if bundle is not None:
            bundle.close()
        if sim_model is not None:
            sim_model.close()
        ros.close()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
