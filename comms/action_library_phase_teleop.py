#!/usr/bin/env python3
"""Follow action-library poses or trajectories from live MediaPipe input.

The default ``live-pose`` mode continuously maps the current hand pose to the
nearest recorded action/phase and supports reversing or changing actions.  The
older ``one-way-sequence`` mode remains available for causal token recognition.
The runner starts DISARMED.  Hardware needs ``HW_ENABLE_TOKEN=1``,
``--enable-motion``, a passing ROS preflight, and a human SPACE press.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from src.comms.action_library import (
    ACTIVE_IDX,
    ActionLibrary,
    G20_ACTION16_TO_CMD20,
    G20_OPEN_POSE,
    G20_SIM_LOWER16,
    G20_SIM_UPPER16,
    JOINT_COUNT,
    LivePoseMatcher,
    OnlinePhaseMatcher,
    PHASE_MAPPING_THUMB_LITTLE_ROUNDTRIP,
    PhaseMatchResult,
    Primitive,
    RESERVED_IDX,
    THUMB_IDX,
    THUMB_LITTLE_DISTANCE_FEATURE_INDEX,
    primitive_template_phase_axis,
    thumb_roundtrip_trajectory,
)
from src.comms.group_action_recorder import draw_hand_overlay
from src.comms.replay_action_group import playback_trajectory
from src.perception.mediapipe_source import MediaPipeHandSource
from src.perception.pipeline import HandPipeline


NONTHUMB_BASE_IDX = (1, 2, 3, 4)
NONTHUMB_SPREAD_IDX = (6, 7, 8, 9)
NONTHUMB_TIP_IDX = (16, 17, 18, 19)
A2_PRIMITIVE_ID = 2
A3_PRIMITIVE_ID = 3
A4_PRIMITIVE_ID = 4
A4_THUMB_ORIENTATION_IDX = (0, 5, 10)


def _parse_four_floats(value: str) -> tuple[float, float, float, float]:
    values = tuple(float(item.strip()) for item in value.split(","))
    if len(values) != 4 or not all(np.isfinite(values)):
        raise argparse.ArgumentTypeError("expected four finite comma-separated values")
    return values


def nonthumb_radians_to_g20_target(
    joint_rad: Sequence[float],
    *,
    base_gain: float = 1.80,
    base_gains: Sequence[float] = (0.72, 1.05, 1.05, 0.63),
    tip_gain: float = 0.80,
    tip_gains: Sequence[float] = (1.34, 0.85, 1.20, 1.27),
) -> np.ndarray:
    """Map only four-finger flexion from retarget radians into G20 range.

    Thumb and spread channels deliberately remain at the G20 open constants;
    callers must use them only through :func:`hybrid_finger_target`, which
    copies those channels exactly from the action-library pose.
    """
    radians = np.asarray(joint_rad, dtype=np.float32).reshape(-1)
    if radians.shape != (JOINT_COUNT,) or not np.all(np.isfinite(radians)):
        raise ValueError("joint_rad must contain 20 finite values")
    base_per_finger = tuple(float(value) for value in base_gains)
    tip_per_finger = tuple(float(value) for value in tip_gains)
    if len(base_per_finger) != 4 or len(tip_per_finger) != 4:
        raise ValueError("base_gains and tip_gains must contain four values")

    command_to_action = {
        command_index: action_index
        for action_index, command_index in enumerate(G20_ACTION16_TO_CMD20)
    }
    target = G20_OPEN_POSE.copy()
    for command_index, gain in zip(
        NONTHUMB_BASE_IDX, base_per_finger
    ):
        action_index = command_to_action[command_index]
        lo = float(G20_SIM_LOWER16[action_index])
        hi = float(G20_SIM_UPPER16[action_index])
        value = float(np.clip(radians[command_index] * float(base_gain) * gain, lo, hi))
        target[command_index] = G20_OPEN_POSE[command_index] - (
            (value - lo) / max(1e-6, hi - lo) * G20_OPEN_POSE[command_index]
        )
    for command_index, gain in zip(
        NONTHUMB_TIP_IDX, tip_per_finger
    ):
        action_index = command_to_action[command_index]
        lo = float(G20_SIM_LOWER16[action_index])
        hi = float(G20_SIM_UPPER16[action_index])
        value = float(np.clip(radians[command_index] * float(tip_gain) * gain, lo, hi))
        target[command_index] = G20_OPEN_POSE[command_index] - (
            (value - lo) / max(1e-6, hi - lo) * G20_OPEN_POSE[command_index]
        )
    target[list(RESERVED_IDX)] = 255.0
    return np.clip(target, 0.0, 255.0)


def full_mediapipe_g20_target(joint_rad: Sequence[float]) -> np.ndarray:
    """Map every active G20 channel directly from the MediaPipe retargeter.

    This bypasses action-library poses but retains the calibrated G20 mapping
    and conservative real-hardware range guard.
    """
    radians = np.asarray(joint_rad, dtype=np.float32).reshape(-1)
    if radians.shape != (JOINT_COUNT,) or not np.all(np.isfinite(radians)):
        raise ValueError("joint_rad must contain 20 finite values")

    sim = np.asarray(
        [radians[index] for index in G20_ACTION16_TO_CMD20], dtype=np.float32
    )
    action_index = {
        command_index: index
        for index, command_index in enumerate(G20_ACTION16_TO_CMD20)
    }
    gains = np.ones(16, dtype=np.float32)
    for command_index, gain in zip(
        NONTHUMB_BASE_IDX, (0.72, 1.05, 1.05, 0.63)
    ):
        gains[action_index[command_index]] = 1.80 * gain
    for command_index, sign in zip(
        NONTHUMB_SPREAD_IDX, (0.35, 1.00, -0.15, -1.00)
    ):
        location = action_index[command_index]
        gains[location] = 0.64 * sign
    for command_index, gain in zip(
        NONTHUMB_TIP_IDX, (1.34, 0.85, 1.20, 1.27)
    ):
        gains[action_index[command_index]] = 0.80 * gain
    gains[action_index[0]] = 0.69
    gains[action_index[5]] = 0.72
    gains[action_index[10]] = 0.96
    gains[action_index[15]] = 1.35
    sim = np.clip(sim * gains, G20_SIM_LOWER16, G20_SIM_UPPER16)

    open_active = G20_OPEN_POSE[list(G20_ACTION16_TO_CMD20)]
    active = open_active - (
        (sim - G20_SIM_LOWER16)
        / (G20_SIM_UPPER16 - G20_SIM_LOWER16)
        * open_active
    )
    for location in (6, 7, 8, 9):
        active[location] = (
            open_active[location]
            + sim[location] / G20_SIM_UPPER16[location] * 100.0
        )

    target = G20_OPEN_POSE.copy()
    target[list(G20_ACTION16_TO_CMD20)] = active
    for command_index, offset in ((0, 20), (5, -28), (10, -24), (15, -27)):
        target[command_index] += offset

    # Same limited-mode corridor used by the prior direct-hardware command.
    for command_index, maximum_delta in ((0, 165), (5, 240), (10, 235)):
        opened = G20_OPEN_POSE[command_index]
        target[command_index] = np.clip(
            target[command_index], opened - maximum_delta, opened + maximum_delta
        )
    flexion = (*NONTHUMB_BASE_IDX, *NONTHUMB_TIP_IDX)
    closure = max(
        float(np.clip(
            (G20_OPEN_POSE[index] - target[index])
            / max(1.0, G20_OPEN_POSE[index]),
            0.0,
            1.0,
        ))
        for index in flexion
    )
    if closure > 0.82:
        alpha = np.clip((closure - 0.82) / 0.18, 0.0, 1.0) * 0.10
        for index in NONTHUMB_SPREAD_IDX:
            target[index] = (
                (1.0 - alpha) * target[index]
                + alpha * G20_OPEN_POSE[index]
            )
    for index in NONTHUMB_SPREAD_IDX:
        opened = G20_OPEN_POSE[index]
        target[index] = np.clip(target[index], opened - 90, opened + 90)
    for left, right in zip(NONTHUMB_SPREAD_IDX, NONTHUMB_SPREAD_IDX[1:]):
        target[left] = max(target[left], target[right])
    for right, left in zip(
        reversed(NONTHUMB_SPREAD_IDX[1:]), reversed(NONTHUMB_SPREAD_IDX[:-1])
    ):
        target[right] = min(target[right], target[left])
    target[list(RESERVED_IDX)] = 255.0
    return np.clip(np.rint(target), 0.0, 255.0).astype(np.float32)


def action_thumb_mediapipe_fingers_target(
    action_target: Sequence[float],
    mediapipe_target: Sequence[float],
) -> np.ndarray:
    """Keep the action's four thumb channels and use MediaPipe everywhere else."""
    action = np.asarray(action_target, dtype=np.float32).reshape(-1)
    mediapipe = np.asarray(mediapipe_target, dtype=np.float32).reshape(-1)
    if action.shape != (JOINT_COUNT,) or mediapipe.shape != (JOINT_COUNT,):
        raise ValueError("action and MediaPipe targets must contain 20 values")
    if not np.all(np.isfinite(action)) or not np.all(np.isfinite(mediapipe)):
        raise ValueError("action and MediaPipe targets must be finite")
    output = mediapipe.copy()
    output[list(THUMB_IDX)] = action[list(THUMB_IDX)]
    output[list(RESERVED_IDX)] = 255.0
    return np.clip(output, 0.0, 255.0)


def frozen_thumb_target(
    target: Sequence[float],
    thumb_hold_pose: Sequence[float],
) -> np.ndarray:
    """Keep the requested non-thumb target while pinning all thumb channels."""
    desired = np.asarray(target, dtype=np.float32).reshape(-1)
    held = np.asarray(thumb_hold_pose, dtype=np.float32).reshape(-1)
    if desired.shape != (JOINT_COUNT,) or held.shape != (JOINT_COUNT,):
        raise ValueError("target and thumb hold pose must contain 20 values")
    if not np.all(np.isfinite(desired)) or not np.all(np.isfinite(held)):
        raise ValueError("target and thumb hold pose must be finite")
    output = desired.copy()
    output[list(THUMB_IDX)] = held[list(THUMB_IDX)]
    output[list(RESERVED_IDX)] = 255.0
    return np.clip(output, 0.0, 255.0)


def offset_thumb_pose(
    current_pose: Sequence[float],
    offsets: Sequence[float],
) -> np.ndarray:
    """Offset q0/q5/q10/q15 while preserving every non-thumb channel."""
    current = np.asarray(current_pose, dtype=np.float32).reshape(-1)
    delta = np.asarray(offsets, dtype=np.float32).reshape(-1)
    if current.shape != (JOINT_COUNT,) or delta.shape != (len(THUMB_IDX),):
        raise ValueError("current pose must contain 20 values and offsets four values")
    if not np.all(np.isfinite(current)) or not np.all(np.isfinite(delta)):
        raise ValueError("current pose and thumb offsets must be finite")
    output = current.copy()
    output[list(THUMB_IDX)] = np.clip(
        current[list(THUMB_IDX)] + delta,
        0.0,
        255.0,
    )
    output[list(RESERVED_IDX)] = 255.0
    return output


def absolute_thumb_pose(
    current_pose: Sequence[float],
    thumb_values: Sequence[float],
) -> np.ndarray:
    """Set q0/q5/q10/q15 absolutely while preserving non-thumb channels."""
    current = np.asarray(current_pose, dtype=np.float32).reshape(-1)
    values = np.asarray(thumb_values, dtype=np.float32).reshape(-1)
    if current.shape != (JOINT_COUNT,) or values.shape != (len(THUMB_IDX),):
        raise ValueError(
            "current pose must contain 20 values and thumb pose four values"
        )
    if not np.all(np.isfinite(current)) or not np.all(np.isfinite(values)):
        raise ValueError("current pose and absolute thumb pose must be finite")
    output = current.copy()
    output[list(THUMB_IDX)] = np.clip(values, 0.0, 255.0)
    output[list(RESERVED_IDX)] = 255.0
    return output


def hybrid_finger_target(
    library_target: Sequence[float],
    mediapipe_target: Sequence[float],
    *,
    base_blend: float,
    tip_blend: float,
    base_residual_limit: float,
    tip_residual_limit: float,
) -> np.ndarray:
    """Add bounded MediaPipe variation only to four-finger flexion channels."""
    library = np.asarray(library_target, dtype=np.float32).reshape(-1)
    mediapipe = np.asarray(mediapipe_target, dtype=np.float32).reshape(-1)
    if library.shape != (JOINT_COUNT,) or mediapipe.shape != (JOINT_COUNT,):
        raise ValueError("hybrid targets must contain 20 values")
    if not np.all(np.isfinite(library)) or not np.all(np.isfinite(mediapipe)):
        raise ValueError("hybrid targets must be finite")
    output = library.copy()
    for indices, blend, limit in (
        (NONTHUMB_BASE_IDX, base_blend, base_residual_limit),
        (NONTHUMB_TIP_IDX, tip_blend, tip_residual_limit),
    ):
        alpha = float(np.clip(blend, 0.0, 1.0))
        residual = alpha * (mediapipe[list(indices)] - library[list(indices)])
        residual = np.clip(residual, -max(0.0, float(limit)), max(0.0, float(limit)))
        output[list(indices)] += residual
    # Pin every excluded channel explicitly: no MediaPipe thumb or spread.
    output[list(THUMB_IDX)] = library[list(THUMB_IDX)]
    output[list(NONTHUMB_SPREAD_IDX)] = library[list(NONTHUMB_SPREAD_IDX)]
    output[list(RESERVED_IDX)] = 255.0
    return np.clip(output, 0.0, 255.0)


def mediapipe_finger_fallback_target(
    library_anchor: Sequence[float],
    mediapipe_target: Sequence[float],
) -> np.ndarray:
    """Use direct MediaPipe flexion while holding library-only channels.

    This is the out-of-library fallback for ``hybrid-fingers`` mode.  It keeps
    the latest library (or arm-time) thumb and spread values, while allowing
    all four non-thumb flexion pairs to cover the full calibrated MediaPipe
    range.  Normal command-step and measured-state-lead guards are still
    applied by the caller.
    """
    anchor = np.asarray(library_anchor, dtype=np.float32).reshape(-1)
    mediapipe = np.asarray(mediapipe_target, dtype=np.float32).reshape(-1)
    if anchor.shape != (JOINT_COUNT,) or mediapipe.shape != (JOINT_COUNT,):
        raise ValueError("fallback targets must contain 20 values")
    if not np.all(np.isfinite(anchor)) or not np.all(np.isfinite(mediapipe)):
        raise ValueError("fallback targets must be finite")

    output = anchor.copy()
    flexion = (*NONTHUMB_BASE_IDX, *NONTHUMB_TIP_IDX)
    output[list(flexion)] = mediapipe[list(flexion)]
    output[list(THUMB_IDX)] = anchor[list(THUMB_IDX)]
    output[list(NONTHUMB_SPREAD_IDX)] = anchor[list(NONTHUMB_SPREAD_IDX)]
    output[list(RESERVED_IDX)] = 255.0
    return np.clip(output, 0.0, 255.0)


def trajectory_target(trajectory: np.ndarray, phase: float) -> np.ndarray:
    """Linearly sample a ``(T,20)`` trajectory at normalized phase."""
    values = np.asarray(trajectory, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != JOINT_COUNT or not len(values):
        raise ValueError(f"trajectory must be non-empty (T,20), got {values.shape}")
    position = float(np.clip(phase, 0.0, 1.0)) * (len(values) - 1)
    left = int(np.floor(position))
    right = min(len(values) - 1, left + 1)
    alpha = position - left
    target = (1.0 - alpha) * values[left] + alpha * values[right]
    target = target.copy()
    target[list(RESERVED_IDX)] = 255.0
    return target


def trajectory_suffix(trajectory: np.ndarray, phase: float) -> np.ndarray:
    """Return the exact current-phase pose followed by the remaining frames."""
    values = np.asarray(trajectory, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != JOINT_COUNT or not len(values):
        raise ValueError(f"trajectory must be non-empty (T,20), got {values.shape}")
    position = float(np.clip(phase, 0.0, 1.0)) * (len(values) - 1)
    current = trajectory_target(values, phase)
    next_index = int(np.floor(position)) + 1
    frames = [current]
    if next_index < len(values):
        frames.extend(values[next_index:])
    suffix = np.asarray(frames, dtype=np.float32)
    suffix[:, list(RESERVED_IDX)] = 255.0
    return suffix


def nearest_trajectory_suffix(
    trajectory: np.ndarray,
    current_pose: Sequence[float],
) -> tuple[np.ndarray, int, float]:
    """Return the nearest recorded frame and everything after it.

    Distance is the RMS command error over the 16 active G20 channels.  The
    four reserved command slots are deliberately ignored.  Ties select the
    earliest frame so a repeated pose retains as much of the action as
    possible.
    """
    values = np.asarray(trajectory, dtype=np.float32)
    current = np.asarray(current_pose, dtype=np.float32).reshape(-1)
    if values.ndim != 2 or values.shape[1] != JOINT_COUNT or not len(values):
        raise ValueError(f"trajectory must be non-empty (T,20), got {values.shape}")
    if current.shape != (JOINT_COUNT,):
        raise ValueError("current pose must contain 20 values")
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(current)):
        raise ValueError("trajectory and current pose must be finite")

    active = list(ACTIVE_IDX)
    differences = values[:, active] - current[active]
    errors = np.sqrt(np.mean(np.square(differences), axis=1))
    frame_index = int(np.argmin(errors))
    suffix = values[frame_index:].copy()
    suffix[:, list(RESERVED_IDX)] = 255.0
    return suffix, frame_index, float(errors[frame_index])


def selected_manual_trajectory(
    trajectory: np.ndarray,
    current_pose: Sequence[float],
    *,
    force_from_start: bool,
) -> tuple[np.ndarray, int, float]:
    """Select a number-key trajectory, optionally preserving its full prefix."""
    if not force_from_start:
        return nearest_trajectory_suffix(trajectory, current_pose)
    values = np.asarray(trajectory, dtype=np.float32)
    current = np.asarray(current_pose, dtype=np.float32).reshape(-1)
    if values.ndim != 2 or values.shape[1] != JOINT_COUNT or not len(values):
        raise ValueError(f"trajectory must be non-empty (T,20), got {values.shape}")
    if current.shape != (JOINT_COUNT,):
        raise ValueError("current pose must contain 20 values")
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(current)):
        raise ValueError("trajectory and current pose must be finite")
    active = list(ACTIVE_IDX)
    error = float(
        np.sqrt(np.mean(np.square(values[0, active] - current[active])))
    )
    selected = values.copy()
    selected[:, list(RESERVED_IDX)] = 255.0
    return selected, 0, error


def manual_action_starts_from_first_frame(
    primitive: Primitive,
    configured_ids: Sequence[int],
) -> bool:
    """Whether number-key playback must preserve the primitive's full prefix."""
    return bool(
        primitive.manual_from_start or primitive.id in set(configured_ids)
    )


def four_finger_spread_score(landmarks: Sequence[Sequence[float]]) -> float:
    """Return lateral fingertip splay normalized by palm width.

    Hand-base axis 1 is the palm's lateral direction. Measuring only this axis
    avoids treating finger-length or fingertip depth differences as splay.
    """
    points = np.asarray(landmarks, dtype=np.float32)
    if points.shape != (21, 3) or not np.all(np.isfinite(points)):
        raise ValueError("landmarks must be finite (21,3)")
    palm_width = float(np.linalg.norm(points[5] - points[17]))
    if palm_width <= 1e-8:
        raise ValueError("cannot measure finger spread from a degenerate palm")
    lateral_tips = points[[8, 12, 16, 20], 1]
    adjacent = np.abs(np.diff(lateral_tips))
    return float(np.mean(adjacent) / palm_width)


class Action23FingerSpreadRouter:
    """Allow only A2 while four fingers are together."""

    def __init__(
        self,
        *,
        threshold: float = 0.350,
        hysteresis: float = 0.030,
        enabled: bool = True,
    ) -> None:
        if threshold <= 0.0 or hysteresis < 0.0:
            raise ValueError("A3 spread-gate values must be nonnegative")
        self.threshold = float(threshold)
        self.hysteresis = float(hysteresis)
        self.enabled = bool(enabled)
        self.reset()

    def reset(self) -> None:
        self.a3_allowed: Optional[bool] = None
        self.score = 0.0

    def update(self, landmarks: Sequence[Sequence[float]]) -> bool:
        self.score = four_finger_spread_score(landmarks)
        if not self.enabled:
            self.a3_allowed = True
        elif self.a3_allowed is None:
            self.a3_allowed = self.score >= self.threshold
        elif (
            not self.a3_allowed
            and self.score >= self.threshold + self.hysteresis
        ):
            self.a3_allowed = True
        elif (
            self.a3_allowed
            and self.score <= self.threshold - self.hysteresis
        ):
            self.a3_allowed = False
        return bool(self.a3_allowed)

    def excluded_ids(self, available_ids: Sequence[int]) -> set[int]:
        """Exclude every non-A2 primitive while the fingers are together."""
        if self.enabled and self.a3_allowed is False:
            return {
                int(primitive_id)
                for primitive_id in available_ids
                if int(primitive_id) != A2_PRIMITIVE_ID
            }
        return set()

    @property
    def allows_a3_assist(self) -> bool:
        return not self.enabled or self.a3_allowed is True

    @property
    def freezes_four_fingers(self) -> bool:
        """Whether AUTO must use A2's fixed four-finger channels only."""
        return self.enabled and self.a3_allowed is False

    @property
    def label(self) -> str:
        if self.a3_allowed is False:
            return "fingers-together: ONLY-A2"
        if self.a3_allowed is True:
            return "fingers-open: ALL-ACTIONS"
        return "off"


class Action3ContactAssist:
    """Acquire reversible A3 tracking before its shared open pose is distinctive.

    The normal AUTO matcher needs class separation.  A3's right/open endpoint
    is intentionally similar to several other primitives, so forward motion can
    fail to acquire even though cold-start reverse motion from thumb contact is
    unambiguous.  This assist still requires proximity to an A3 template, but
    uses the dedicated thumb-to-little contact feature to activate A3 as soon as
    measurable contact progress begins.
    """

    def __init__(
        self,
        primitive: Primitive,
        *,
        activate_phase: float = 0.08,
        release_phase: float = 0.02,
        confirm_frames: int = 2,
        reject_frames: int = 4,
        threshold_scale: float = 1.60,
        max_phase_step: float = 0.18,
        phase_smoothing: float = 0.65,
        competitors: Sequence[Primitive] = (),
        competition_slack: float = 0.015,
        enabled: bool = True,
    ) -> None:
        if not 0.0 <= release_phase < activate_phase <= 1.0:
            raise ValueError("A3 contact phases must satisfy 0 <= release < activate <= 1")
        if (
            confirm_frames <= 0
            or reject_frames <= 0
            or threshold_scale <= 0.0
            or competition_slack < 0.0
        ):
            raise ValueError("A3 contact frame counts and threshold scale must be positive")
        self.primitive = primitive
        self.activate_phase = float(activate_phase)
        self.release_phase = float(release_phase)
        self.confirm_frames = int(confirm_frames)
        self.reject_frames = int(reject_frames)
        self.threshold_scale = float(threshold_scale)
        self.max_phase_step = max(1e-6, float(max_phase_step))
        self.phase_smoothing = float(np.clip(phase_smoothing, 0.0, 1.0))
        self.competitors = tuple(competitors)
        self.competition_slack = float(competition_slack)
        self.enabled = bool(enabled)
        self.roundtrip = (
            primitive.phase_mapping == PHASE_MAPPING_THUMB_LITTLE_ROUNDTRIP
        )
        self.phase_continuity_penalty = 0.010
        self._phase_axes = tuple(
            primitive_template_phase_axis(primitive, template)
            for template in primitive.templates
        )
        if self.roundtrip:
            contact_values = np.asarray([
                float(np.min(template[:, THUMB_LITTLE_DISTANCE_FEATURE_INDEX]))
                for template in primitive.templates
            ])
            self.roundtrip_start_distance = float(np.median([
                template[0, THUMB_LITTLE_DISTANCE_FEATURE_INDEX]
                for template in primitive.templates
            ]))
            self.roundtrip_contact_distance = float(np.median(contact_values))
            self.roundtrip_contact_gate = float(np.max(contact_values) + 0.03)
            self.roundtrip_return_distance = float(np.min([
                template[-1, THUMB_LITTLE_DISTANCE_FEATURE_INDEX]
                for template in primitive.templates
            ]))
        self.reset()

    def reset(self) -> None:
        self.active = False
        self.phase = 0.0
        self.candidate_frames = 0
        self.release_frames = 0
        self.rejected_frames = 0
        self.roundtrip_branch = "inward"

    def _score(self, value: np.ndarray) -> tuple[float, float, float]:
        candidates: list[tuple[float, float, float]] = []
        for template, phases in zip(self.primitive.templates, self._phase_axes):
            if template.shape[1] != value.shape[0]:
                raise ValueError("A3 contact feature width does not match template")
            distances = np.sqrt(np.mean((template - value[None, :]) ** 2, axis=1))
            adjusted = distances
            if self.active:
                adjusted = distances + self.phase_continuity_penalty * np.abs(
                    phases - self.phase
                )
            index = int(np.argmin(adjusted))
            candidates.append((
                float(adjusted[index]), float(distances[index]), float(phases[index])
            ))
        _adjusted, distance, phase = min(candidates, key=lambda item: item[0])
        if self.roundtrip:
            contact_value = float(value[THUMB_LITTLE_DISTANCE_FEATURE_INDEX])
            if contact_value <= self.roundtrip_contact_gate:
                self.roundtrip_branch = "outward"
            contact_phase = self.primitive.phase_contact_fraction
            if self.roundtrip_branch == "inward":
                span = max(
                    1e-6,
                    self.roundtrip_start_distance
                    - self.roundtrip_contact_distance,
                )
                phase = contact_phase * np.clip(
                    (self.roundtrip_start_distance - contact_value) / span,
                    0.0,
                    1.0,
                )
            else:
                span = max(
                    1e-6,
                    self.roundtrip_return_distance
                    - self.roundtrip_contact_distance,
                )
                phase = contact_phase + (1.0 - contact_phase) * np.clip(
                    (contact_value - self.roundtrip_contact_distance) / span,
                    0.0,
                    1.0,
                )
        competitor_distance = min(
            (
                float(np.min(np.sqrt(np.mean(
                    (template - value[None, :]) ** 2, axis=1
                ))))
                for primitive in self.competitors
                for template in primitive.templates
            ),
            default=np.inf,
        )
        return phase, distance, competitor_distance

    def _smooth(self, raw_phase: float, *, initialize: bool = False) -> None:
        raw = float(np.clip(raw_phase, 0.0, 1.0))
        if initialize:
            self.phase = raw
            return
        delta = float(np.clip(
            raw - self.phase, -self.max_phase_step, self.max_phase_step
        ))
        self.phase = float(np.clip(
            self.phase + self.phase_smoothing * delta, 0.0, 1.0
        ))

    def update(self, feature: Sequence[float]) -> Optional[PhaseMatchResult]:
        if not self.enabled:
            return None
        value = np.asarray(feature, dtype=np.float32).reshape(-1)
        if (
            value.shape[0] <= THUMB_LITTLE_DISTANCE_FEATURE_INDEX
            or not np.all(np.isfinite(value))
        ):
            raise ValueError("A3 contact feature must be finite and include contact distance")
        raw_phase, distance, competitor_distance = self._score(value)
        accepted = (
            distance <= self.primitive.threshold * self.threshold_scale
            and distance <= competitor_distance + self.competition_slack
        )

        if not self.active:
            if accepted and raw_phase >= self.activate_phase:
                self.candidate_frames += 1
            else:
                self.candidate_frames = 0
            if self.candidate_frames < self.confirm_frames:
                return None
            self.active = True
            self.candidate_frames = 0
            self.release_frames = 0
            self.rejected_frames = 0
            self._smooth(raw_phase, initialize=True)
        else:
            self.rejected_frames = 0 if accepted else self.rejected_frames + 1
            if self.rejected_frames >= self.reject_frames:
                self.reset()
                return None
            release_reached = (
                False if self.roundtrip else raw_phase <= self.release_phase
            )
            self.release_frames = (
                self.release_frames + 1
                if accepted and release_reached
                else 0
            )
            self._smooth(raw_phase)
            if self.roundtrip and raw_phase >= 1.0 - 1e-6:
                self.phase = 1.0

        result = PhaseMatchResult(
            primitive_id=self.primitive.id,
            name=f"{self.primitive.name}/contact_assist",
            phase=float(self.phase),
            distance=float(distance),
            second_distance=float(competitor_distance),
            locked=True,
        )
        if self.release_frames >= self.confirm_frames:
            self.reset()
        return result


def action3_assist_may_acquire(
    previous: Optional[PhaseMatchResult],
) -> bool:
    """Do not let A3 steal control from an already locked non-A3 action."""
    return bool(
        previous is None
        or not previous.locked
        or previous.primitive_id == A3_PRIMITIVE_ID
    )


def step_limited_command(
    target: Sequence[float], base: Sequence[float], *, max_step: int
) -> list[int]:
    """Move active SDK-range channels from measured base toward target."""
    desired = np.asarray(target, dtype=np.float32).reshape(-1)
    current = np.asarray(base, dtype=np.float32).reshape(-1)
    if desired.shape != (JOINT_COUNT,) or current.shape != (JOINT_COUNT,):
        raise ValueError("target and base must contain 20 values")
    if not np.all(np.isfinite(desired)) or not np.all(np.isfinite(current)):
        raise ValueError("target and base must be finite")
    step = max(1, int(max_step))
    command = current.copy()
    active = list(ACTIVE_IDX)
    command[active] += np.clip(desired[active] - current[active], -step, step)
    command = np.clip(np.rint(command), 0, 255).astype(np.int32)
    command[list(RESERVED_IDX)] = 255
    return command.tolist()


def state_guarded_command(
    target: Sequence[float],
    previous: Sequence[float],
    observed: Optional[Sequence[float]],
    *,
    max_step: int,
    max_state_lead: int,
) -> list[int]:
    """Slew from the previous command while bounding lead over measured state.

    Stepping from measured state on every frame keeps the command only one step
    ahead of the motor and makes tracking unnecessarily slow.  Stepping from
    the previous command preserves the requested command rate; the state-lead
    guard still prevents an accumulating command backlog.
    """
    command = np.asarray(
        step_limited_command(target, previous, max_step=max_step),
        dtype=np.float32,
    )
    if observed is not None:
        state = np.asarray(observed, dtype=np.float32).reshape(-1)
        if state.shape != (JOINT_COUNT,) or not np.all(np.isfinite(state)):
            raise ValueError("observed state must contain 20 finite values")
        lead = max(1, int(max_state_lead))
        active = list(ACTIVE_IDX)
        command[active] = np.clip(
            command[active], state[active] - lead, state[active] + lead
        )
    command = np.clip(np.rint(command), 0, 255).astype(np.int32)
    command[list(RESERVED_IDX)] = 255
    return command.tolist()


class Action4ThumbTipGate:
    """Enforce A4 left-align -> tip-close -> right-turn ordering.

    The first A4 library pose combines the desired left-aligned orientation
    with a closed q15.  Sending it directly makes both motions start together;
    the former gate did the opposite and froze orientation while closing q15.
    This state machine first drives q0/q5/q10 to that left pose while holding
    q15 at its entry value, then closes q15 while holding the left pose, and
    only then releases the right turn and remaining trajectory.
    """

    def __init__(
        self,
        *,
        tolerance: float,
        confirm_frames: int,
        left_tolerance: float = 5.0,
        left_confirm_frames: int = 3,
        enabled: bool = True,
    ):
        self.tolerance = float(tolerance)
        self.confirm_frames = int(confirm_frames)
        self.left_tolerance = float(left_tolerance)
        self.left_confirm_frames = int(left_confirm_frames)
        self.enabled = bool(enabled)
        self.reset()

    def reset(self) -> None:
        self.active = False
        self.released = False
        self.confirmed = 0
        self.stage = "idle"
        self.last_applied_stage = "idle"
        self.entry_tip: Optional[float] = None
        self.last_error = float("inf")

    def apply(
        self,
        target: Sequence[float],
        *,
        primitive_id: Optional[int],
        close_pose: Sequence[float],
        observed: Optional[Sequence[float]],
        previous: Sequence[float],
    ) -> tuple[np.ndarray, bool, bool]:
        """Return ``(target, waiting, released_now)`` for the active action."""
        desired = np.asarray(target, dtype=np.float32).reshape(-1)
        closed = np.asarray(close_pose, dtype=np.float32).reshape(-1)
        prior = np.asarray(previous, dtype=np.float32).reshape(-1)
        if (
            desired.shape != (JOINT_COUNT,)
            or closed.shape != (JOINT_COUNT,)
            or prior.shape != (JOINT_COUNT,)
            or not np.all(np.isfinite(desired))
            or not np.all(np.isfinite(closed))
            or not np.all(np.isfinite(prior))
        ):
            raise ValueError("A4 gate poses must contain 20 finite values")

        if not self.enabled or primitive_id != A4_PRIMITIVE_ID:
            self.reset()
            output = desired.copy()
            output[list(RESERVED_IDX)] = 255.0
            return output, False, False

        feedback = prior
        if observed is not None:
            feedback = np.asarray(observed, dtype=np.float32).reshape(-1)
            if feedback.shape != (JOINT_COUNT,) or not np.all(np.isfinite(feedback)):
                raise ValueError("A4 gate feedback must contain 20 finite values")

        if not self.active:
            self.active = True
            self.stage = "left"
            self.entry_tip = float(feedback[15])

        if self.released:
            output = desired.copy()
            output[list(RESERVED_IDX)] = 255.0
            return output, False, False

        if self.stage == "left":
            self.last_applied_stage = "left"
            self.last_error = float(
                np.max(
                    np.abs(
                        feedback[list(A4_THUMB_ORIENTATION_IDX)]
                        - closed[list(A4_THUMB_ORIENTATION_IDX)]
                    )
                )
            )
            if self.last_error <= self.left_tolerance:
                self.confirmed += 1
            else:
                self.confirmed = 0
            output = desired.copy()
            output[list(A4_THUMB_ORIENTATION_IDX)] = closed[
                list(A4_THUMB_ORIENTATION_IDX)
            ]
            assert self.entry_tip is not None
            output[15] = self.entry_tip
            output[list(RESERVED_IDX)] = 255.0
            if self.confirmed >= self.left_confirm_frames:
                self.stage = "tip"
                self.confirmed = 0
            return np.clip(output, 0.0, 255.0), True, False

        if self.stage == "tip":
            self.last_applied_stage = "tip"
            self.last_error = abs(float(feedback[15]) - float(closed[15]))
            if self.last_error <= self.tolerance:
                self.confirmed += 1
            else:
                self.confirmed = 0
            if self.confirmed >= self.confirm_frames:
                self.released = True
                self.stage = "released"
                output = desired.copy()
                # Keep q15 closed on the exact frame right rotation releases.
                output[15] = closed[15]
                output[list(RESERVED_IDX)] = 255.0
                return output, False, True

            output = desired.copy()
            output[list(A4_THUMB_ORIENTATION_IDX)] = closed[
                list(A4_THUMB_ORIENTATION_IDX)
            ]
            output[15] = closed[15]
            output[list(RESERVED_IDX)] = 255.0
            return np.clip(output, 0.0, 255.0), True, False

        raise RuntimeError(f"unexpected A4 gate stage {self.stage!r}")


def active_pose_error(
    observed: Sequence[float], target: Sequence[float]
) -> float:
    """Return the largest absolute error across non-reserved G20 channels."""
    state = np.asarray(observed, dtype=np.float32).reshape(-1)
    desired = np.asarray(target, dtype=np.float32).reshape(-1)
    if state.shape != (JOINT_COUNT,) or desired.shape != (JOINT_COUNT,):
        raise ValueError("observed and target must contain 20 values")
    if not np.all(np.isfinite(state)) or not np.all(np.isfinite(desired)):
        raise ValueError("observed and target must be finite")
    return float(
        np.max(np.abs(state[list(ACTIVE_IDX)] - desired[list(ACTIVE_IDX)]))
    )


def reset_completion(
    arm_after_reset: bool, pose_label: str = "open pose"
) -> tuple[bool, str]:
    """Return the arm state/message after an episode-start reset settles."""
    if arm_after_reset:
        return True, "RESET COMPLETE; ARMED and ACT episode started"
    return False, f"RESET COMPLETE; DISARMED at {pose_label}"


def is_delete_last_episode_key(key: int) -> bool:
    """Accept D/d only; R is reserved for manual reset."""
    return key in (ord("d"), ord("D"))


def is_manual_reset_key(key: int) -> bool:
    """Accept R/r as the explicit reset-without-recording key."""
    return key in (ord("r"), ord("R"))


def is_full_teleop_toggle_key(key: int) -> bool:
    """Accept T/t as the full-MediaPipe/AUTO mode toggle."""
    return key in (ord("t"), ord("T"))


def is_thumb_freeze_toggle_key(key: int) -> bool:
    """Accept F/f as the current-DexHand-thumb hold toggle."""
    return key in (ord("f"), ord("F"))


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--camera-index", type=int, default=2)
    parser.add_argument("--side", choices=("right", "left"), default="right")
    parser.add_argument("--rate", type=float, default=30.0)
    parser.add_argument(
        "--tracking-mode",
        choices=("live-pose", "one-way-sequence"),
        default="live-pose",
        help="live-pose continuously follows/reverses; one-way-sequence locks one token",
    )
    parser.add_argument(
        "--control-mode",
        choices=("library", "hybrid-fingers"),
        default="library",
        help=(
            "library only, or bounded finger residual while matched plus direct "
            "four-finger MediaPipe fallback while unmatched"
        ),
    )
    parser.add_argument(
        "--hybrid-unlocked-fingers",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "in hybrid mode, keep four-finger flexion responsive to MediaPipe "
            "when no action-library pose is locked"
        ),
    )
    parser.add_argument(
        "--primitive-id",
        type=int,
        default=None,
        help="live-pose only: force one action ID for immediate, unambiguous tracking",
    )
    parser.add_argument("--min-detection-confidence", type=float, default=0.75)
    parser.add_argument("--min-tracking-confidence", type=float, default=0.75)
    parser.add_argument("--min-hand-score", type=float, default=0.5)
    parser.add_argument("--hand-lost-frames", type=int, default=5)
    parser.add_argument("--min-lock-phase", type=float, default=0.20)
    parser.add_argument("--lock-margin", type=float, default=0.015)
    parser.add_argument("--lock-confirm-frames", type=int, default=2)
    parser.add_argument("--threshold-scale", type=float, default=1.20)
    parser.add_argument("--max-template-advance", type=int, default=3)
    parser.add_argument("--max-phase-step", type=float, default=0.18)
    parser.add_argument("--phase-score-slack", type=float, default=0.012)
    parser.add_argument("--phase-smoothing", type=float, default=0.65)
    parser.add_argument("--switch-margin", type=float, default=0.015)
    parser.add_argument("--switch-confirm-frames", type=int, default=2)
    parser.add_argument(
        "--a23-spread-routing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="prefer A2 for together fingers and A3 for spread fingers",
    )
    parser.add_argument("--a23-spread-threshold", type=float, default=0.350)
    parser.add_argument("--a23-spread-hysteresis", type=float, default=0.030)
    parser.add_argument(
        "--a3-contact-assist",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="direction-symmetric early acquisition for thumb-to-little action 3",
    )
    parser.add_argument("--a3-contact-activate-phase", type=float, default=0.08)
    parser.add_argument("--a3-contact-release-phase", type=float, default=0.02)
    parser.add_argument("--a3-contact-confirm-frames", type=int, default=2)
    parser.add_argument("--a3-contact-threshold-scale", type=float, default=1.60)
    parser.add_argument("--a3-contact-competition-slack", type=float, default=0.015)
    parser.add_argument("--finger-base-blend", type=float, default=0.15)
    parser.add_argument("--finger-tip-blend", type=float, default=0.20)
    parser.add_argument("--finger-base-residual-limit", type=float, default=20.0)
    parser.add_argument("--finger-tip-residual-limit", type=float, default=25.0)
    parser.add_argument("--mediapipe-base-gain", type=float, default=1.80)
    parser.add_argument(
        "--mediapipe-base-gains",
        type=_parse_four_floats,
        default=(0.72, 1.05, 1.05, 0.63),
    )
    parser.add_argument("--mediapipe-tip-gain", type=float, default=0.80)
    parser.add_argument(
        "--mediapipe-tip-gains",
        type=_parse_four_floats,
        default=(1.34, 0.85, 1.20, 1.27),
    )
    parser.add_argument("--max-range-step", type=int, default=5)
    parser.add_argument("--max-state-lead", type=int, default=30)
    parser.add_argument(
        "--manual-blend-frames",
        type=int,
        default=8,
        help="number-key replay: minimum transition frames into the action start",
    )
    parser.add_argument(
        "--a4-thumb-tip-gate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "enforce measured A4 left-align, then q15 close, then right-turn "
            "ordering"
        ),
    )
    parser.add_argument(
        "--a4-thumb-tip-tolerance",
        type=float,
        default=5.0,
        help="maximum measured q15 error before A4 rotation may be released",
    )
    parser.add_argument(
        "--a4-thumb-tip-confirm-frames",
        type=int,
        default=3,
        help="consecutive in-tolerance feedback frames required before A4 rotation",
    )
    parser.add_argument(
        "--a4-left-align-tolerance",
        type=float,
        default=5.0,
        help="maximum q0/q5/q10 error before A4 may begin closing q15",
    )
    parser.add_argument(
        "--a4-left-align-confirm-frames",
        type=int,
        default=3,
        help="consecutive left-alignment feedback frames required before tip closure",
    )
    parser.add_argument(
        "--reset-before-arm",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="after disarmed SPACE, settle at the episode start pose before recording",
    )
    parser.add_argument(
        "--reset-on-start",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="once ready, move to the episode start pose and stay disarmed",
    )
    parser.add_argument(
        "--reset-after-disarm",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="after armed SPACE stops recording, return to the episode start pose",
    )
    parser.add_argument(
        "--episode-start-action-end",
        type=int,
        default=None,
        help=(
            "use this action's final library frame instead of open pose for "
            "startup/before-arm/after-disarm resets; recorder stays stopped"
        ),
    )
    parser.add_argument(
        "--manual-action-from-start",
        type=int,
        action="append",
        default=[],
        help=(
            "number-key action ID that must replay from frame 1 instead of "
            "the nearest current-pose frame; repeat for multiple IDs"
        ),
    )
    parser.add_argument(
        "--freeze-thumb-on-start",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "after the first fresh hand state, latch its q0/q5/q10/q15 values; "
            "F still releases or re-latches the thumb"
        ),
    )
    parser.add_argument(
        "--startup-thumb-offsets",
        type=_parse_four_floats,
        default=None,
        metavar="DQ0,DQ5,DQ10,DQ15",
        help=(
            "with --freeze-thumb-on-start and --reset-on-start, move the "
            "measured thumb by these four SDK-tick offsets before latching it"
        ),
    )
    parser.add_argument(
        "--startup-thumb-pose",
        type=_parse_four_floats,
        default=None,
        metavar="Q0,Q5,Q10,Q15",
        help=(
            "with --freeze-thumb-on-start and --reset-on-start, move to these "
            "absolute q0/q5/q10/q15 SDK values before latching the thumb"
        ),
    )
    parser.add_argument(
        "--thumb-roundtrip-key",
        type=int,
        default=6,
        help=(
            "number key for a dynamic thumb-only nearest-frame roundtrip; "
            "0 disables this override"
        ),
    )
    parser.add_argument(
        "--thumb-roundtrip-source-action",
        type=int,
        default=2,
        help="action whose thumb path supplies the dynamic roundtrip",
    )
    parser.add_argument("--reset-tolerance", type=float, default=12.0)
    parser.add_argument("--reset-timeout", type=float, default=5.0)
    parser.add_argument("--reset-confirm-frames", type=int, default=3)
    parser.add_argument("--current-limit", type=int, default=20)
    parser.add_argument("--speed-limit", type=int, default=35)
    parser.add_argument("--command-timeout", type=float, default=5.0)
    parser.add_argument("--state-timeout", type=float, default=5.0)
    parser.add_argument("--state-stale-seconds", type=float, default=0.5)
    parser.add_argument("--require-state", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--minimal-overlay",
        action="store_true",
        help=(
            "show only arm state and hand/mode/phase; hide router, status, "
            "and key-help text"
        ),
    )
    parser.add_argument("--enable-motion", action="store_true")
    return parser.parse_args(argv)


def _overlay(
    frame: np.ndarray,
    *,
    armed: bool,
    fresh: bool,
    status: str,
    locked: bool,
    manual: bool,
    full_teleop: bool,
    thumb_frozen: bool,
    a23_enabled: bool,
    a23_label: str,
    a23_score: float,
    a23_threshold: float,
    resetting: bool,
    reset_before_arm: bool,
    reset_after_disarm: bool,
    phase: float,
    minimal: bool,
) -> None:
    import cv2

    colour = (0, 220, 0) if armed else (0, 180, 255)
    state_label = "RESETTING" if resetting else ("ARMED" if armed else "DISARMED")
    cv2.putText(
        frame,
        state_label,
        (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        colour,
        2,
    )
    mode = (
        "RESETTING"
        if resetting
        else (
            "THUMB HOLD"
            if thumb_frozen
            else (
                "MANUAL"
                if manual
                else ("FULL MP" if full_teleop else ("LOCKED" if locked else "SEARCHING"))
            )
        )
    )
    cv2.putText(
        frame,
        f"hand={'fresh' if fresh else 'lost'}  {mode}  phase={phase:.0%}",
        (15, 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
    )
    if minimal:
        return
    cv2.putText(
        frame,
        (
            f"A23-ROUTER {a23_label}  spread={a23_score:.3f} "
            f"threshold={a23_threshold:.3f}"
            if a23_enabled
            else "A23 finger-spread router OFF"
        ),
        (15, 84),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (0, 80, 255) if "only-a2" in a23_label.lower() else (0, 255, 120),
        1,
    )
    cv2.putText(
        frame,
        status[-96:],
        (15, 108),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (255, 255, 0),
        1,
    )
    space_start = "reset+start" if reset_before_arm else "start"
    space_stop = "stop+reset" if reset_after_disarm else "stop"
    cv2.putText(
        frame,
        f"SPACE {space_start}/{space_stop}   R reset   T mode   F thumb hold   "
        "1-9 action   0 AUTO   D delete last   Q/ESC exit",
        (15, 132),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (255, 255, 255),
        1,
    )


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    if not 0 < args.rate <= 60:
        print("[phase_teleop] --rate must be in (0, 60]", file=sys.stderr)
        return 2
    if args.max_range_step <= 0:
        print("[phase_teleop] --max-range-step must be positive", file=sys.stderr)
        return 2
    if args.max_state_lead <= 0:
        print("[phase_teleop] --max-state-lead must be positive", file=sys.stderr)
        return 2
    if args.manual_blend_frames < 0:
        print("[phase_teleop] --manual-blend-frames must be nonnegative", file=sys.stderr)
        return 2
    if not 0 <= args.thumb_roundtrip_key <= 9:
        print("[phase_teleop] --thumb-roundtrip-key must be in 0..9", file=sys.stderr)
        return 2
    startup_thumb_override = (
        args.startup_thumb_offsets is not None
        or args.startup_thumb_pose is not None
    )
    if (
        args.startup_thumb_offsets is not None
        and args.startup_thumb_pose is not None
    ):
        print(
            "[phase_teleop] choose either --startup-thumb-offsets or "
            "--startup-thumb-pose, not both",
            file=sys.stderr,
        )
        return 2
    if startup_thumb_override:
        if not args.freeze_thumb_on_start:
            print(
                "[phase_teleop] startup thumb override requires "
                "--freeze-thumb-on-start",
                file=sys.stderr,
            )
            return 2
        if not args.reset_on_start:
            print(
                "[phase_teleop] startup thumb override requires "
                "--reset-on-start so the target is reached while DISARMED",
                file=sys.stderr,
            )
            return 2
    if (
        args.a23_spread_threshold <= 0.0
        or args.a23_spread_hysteresis < 0.0
    ):
        print(
            "[phase_teleop] A3 spread threshold must be positive and "
            "hysteresis must be nonnegative",
            file=sys.stderr,
        )
        return 2
    if (
        args.a4_thumb_tip_tolerance < 0.0
        or args.a4_thumb_tip_confirm_frames <= 0
        or args.a4_left_align_tolerance < 0.0
        or args.a4_left_align_confirm_frames <= 0
    ):
        print(
            "[phase_teleop] A4 gate tolerances must be nonnegative and "
            "confirm-frame counts positive",
            file=sys.stderr,
        )
        return 2
    if (
        args.reset_tolerance <= 0.0
        or args.reset_timeout <= 0.0
        or args.reset_confirm_frames <= 0
    ):
        print("[phase_teleop] reset tolerance/timeout/confirm must be positive", file=sys.stderr)
        return 2
    if not 0.0 <= args.phase_smoothing <= 1.0:
        print("[phase_teleop] --phase-smoothing must be in [0, 1]", file=sys.stderr)
        return 2
    if not (
        0.0 <= args.a3_contact_release_phase
        < args.a3_contact_activate_phase <= 1.0
    ):
        print(
            "[phase_teleop] A3 contact phases must satisfy "
            "0 <= release < activate <= 1",
            file=sys.stderr,
        )
        return 2
    if (
        args.a3_contact_confirm_frames <= 0
        or args.a3_contact_threshold_scale <= 0.0
        or args.a3_contact_competition_slack < 0.0
    ):
        print(
            "[phase_teleop] A3 contact confirm/threshold must be positive",
            file=sys.stderr,
        )
        return 2
    if not 0.0 <= args.finger_base_blend <= 1.0 or not 0.0 <= args.finger_tip_blend <= 1.0:
        print("[phase_teleop] finger blend values must be in [0, 1]", file=sys.stderr)
        return 2
    if args.finger_base_residual_limit < 0 or args.finger_tip_residual_limit < 0:
        print("[phase_teleop] finger residual limits must be nonnegative", file=sys.stderr)
        return 2
    mapping_gains = (
        args.mediapipe_base_gain,
        args.mediapipe_tip_gain,
        *args.mediapipe_base_gains,
        *args.mediapipe_tip_gains,
    )
    if not all(np.isfinite(mapping_gains)) or any(value < 0 for value in mapping_gains):
        print("[phase_teleop] MediaPipe finger gains must be finite and nonnegative", file=sys.stderr)
        return 2
    if not 1 <= args.current_limit <= 30:
        print("[phase_teleop] --current-limit must be in conservative range 1..30", file=sys.stderr)
        return 2
    if not 1 <= args.speed_limit <= 50:
        print("[phase_teleop] --speed-limit must be in conservative range 1..50", file=sys.stderr)
        return 2
    if args.state_stale_seconds <= 0:
        print("[phase_teleop] --state-stale-seconds must be positive", file=sys.stderr)
        return 2
    if args.enable_motion and os.environ.get("HW_ENABLE_TOKEN") != "1":
        print(
            "[phase_teleop] refusing hardware: a human must set HW_ENABLE_TOKEN=1",
            file=sys.stderr,
        )
        return 2

    try:
        library = ActionLibrary.load(args.library)
    except (OSError, ValueError, KeyError) as exc:
        print(f"[phase_teleop] cannot load library: {exc}", file=sys.stderr)
        return 2
    if args.primitive_id is not None and args.primitive_id not in library.primitives:
        print(
            f"[phase_teleop] unknown --primitive-id {args.primitive_id}; "
            f"available={sorted(library.primitives)}",
            file=sys.stderr,
        )
        return 2
    if (
        args.thumb_roundtrip_key
        and args.thumb_roundtrip_key not in library.primitives
    ):
        print(
            f"[phase_teleop] thumb roundtrip key {args.thumb_roundtrip_key} "
            f"is not an installed action; available={sorted(library.primitives)}",
            file=sys.stderr,
        )
        return 2
    if (
        args.thumb_roundtrip_key
        and args.thumb_roundtrip_source_action not in library.primitives
    ):
        print(
            f"[phase_teleop] thumb roundtrip source action "
            f"{args.thumb_roundtrip_source_action} is unavailable; "
            f"available={sorted(library.primitives)}",
            file=sys.stderr,
        )
        return 2
    if (
        args.episode_start_action_end is not None
        and args.episode_start_action_end not in library.primitives
    ):
        print(
            f"[phase_teleop] unknown --episode-start-action-end "
            f"{args.episode_start_action_end}; "
            f"available={sorted(library.primitives)}",
            file=sys.stderr,
        )
        return 2
    unknown_manual_from_start = sorted(
        set(args.manual_action_from_start) - set(library.primitives)
    )
    if unknown_manual_from_start:
        print(
            "[phase_teleop] unknown --manual-action-from-start IDs "
            f"{unknown_manual_from_start}; available={sorted(library.primitives)}",
            file=sys.stderr,
        )
        return 2
    if args.episode_start_action_end is None:
        episode_start_pose = G20_OPEN_POSE.copy()
        episode_start_label = "open pose"
    else:
        start_primitive = library.primitives[args.episode_start_action_end]
        episode_start_pose = np.asarray(
            start_primitive.trajectory[-1], dtype=np.float32
        ).copy()
        episode_start_pose[list(RESERVED_IDX)] = 255.0
        episode_start_label = f"action {start_primitive.id} endpoint"
    if args.primitive_id is not None and args.tracking_mode != "live-pose":
        print("[phase_teleop] --primitive-id requires --tracking-mode live-pose", file=sys.stderr)
        return 2
    if args.tracking_mode == "live-pose":
        matcher = LivePoseMatcher(
            library,
            selected_id=args.primitive_id,
            match_margin=args.lock_margin,
            confirm_frames=args.lock_confirm_frames,
            switch_margin=args.switch_margin,
            switch_confirm_frames=args.switch_confirm_frames,
            threshold_scale=args.threshold_scale,
            max_phase_step=args.max_phase_step,
            phase_smoothing=args.phase_smoothing,
        )
    else:
        matcher = OnlinePhaseMatcher(
            library,
            min_lock_phase=args.min_lock_phase,
            lock_margin=args.lock_margin,
            confirm_frames=args.lock_confirm_frames,
            threshold_scale=args.threshold_scale,
            max_template_advance=args.max_template_advance,
            max_phase_step=args.max_phase_step,
            phase_score_slack=args.phase_score_slack,
        )
    a23_spread_router = Action23FingerSpreadRouter(
        threshold=args.a23_spread_threshold,
        hysteresis=args.a23_spread_hysteresis,
        enabled=(
            args.a23_spread_routing
            and args.tracking_mode == "live-pose"
            and A3_PRIMITIVE_ID in library.primitives
        ),
    )
    a3_contact_assist: Optional[Action3ContactAssist] = None
    if args.tracking_mode == "live-pose" and A3_PRIMITIVE_ID in library.primitives:
        a3_contact_assist = Action3ContactAssist(
            library.primitives[A3_PRIMITIVE_ID],
            activate_phase=args.a3_contact_activate_phase,
            release_phase=args.a3_contact_release_phase,
            confirm_frames=args.a3_contact_confirm_frames,
            threshold_scale=args.a3_contact_threshold_scale,
            max_phase_step=args.max_phase_step,
            phase_smoothing=args.phase_smoothing,
            competitors=tuple(
                primitive
                for primitive_id, primitive in library.primitives.items()
                if primitive_id != A3_PRIMITIVE_ID
            ),
            competition_slack=args.a3_contact_competition_slack,
            enabled=(
                args.a3_contact_assist
                and args.primitive_id in (None, A3_PRIMITIVE_ID)
            ),
        )
    from src.finger_retarget import retarget as retarget_fn

    ros = None
    rclpy = None
    hand_source = None
    state_watch = None
    state_clock = {"updated": 0.0}
    last_command: Optional[np.ndarray] = None
    library_anchor: Optional[np.ndarray] = None
    manual_primitive_id: Optional[int] = None
    manual_frames: Optional[np.ndarray] = None
    manual_source_trajectory: Optional[np.ndarray] = None
    manual_frame = 0
    manual_thumb_hybrid = False
    full_teleop = False
    thumb_hold_pose: Optional[np.ndarray] = None
    a4_thumb_tip_gate = Action4ThumbTipGate(
        tolerance=args.a4_thumb_tip_tolerance,
        confirm_frames=args.a4_thumb_tip_confirm_frames,
        left_tolerance=args.a4_left_align_tolerance,
        left_confirm_frames=args.a4_left_align_confirm_frames,
        enabled=args.a4_thumb_tip_gate,
    )
    armed = False
    resetting = False
    arm_after_reset = False
    reset_started = 0.0
    reset_confirmed = 0
    lost_frames = 0
    status = (
        f"loaded {len(library.primitives)} actions; "
        f"feature={library.feature_profile}; mode={args.tracking_mode}/"
        f"{args.control_mode}; SPACE"
    )
    last_log = 0.0
    last_result = None
    try:
        if args.enable_motion:
            import rclpy as _rclpy
            from src.comms.camera_to_linkerhand import L20RosNode

            rclpy = _rclpy
            rclpy.init(args=None)
            ros = L20RosNode(args)
            if not ros.wait_ready():
                return 2
            command_publishers = ros.node.count_publishers(ros.cmd_topic)
            if command_publishers > 1:
                print(
                    f"[phase_teleop] refusing hardware: found {command_publishers} "
                    f"publishers on {ros.cmd_topic}; close the official GUI",
                    file=sys.stderr,
                )
                return 2

            def _state_heartbeat(_message) -> None:
                state_clock["updated"] = time.monotonic()

            state_watch = ros.node.create_subscription(
                ros.JointState, ros.state_topic, _state_heartbeat, 10
            )
            state_clock["updated"] = time.monotonic()
            ros.publish_settings()
            ros.publish_session_active(False)
            if ros.last_state is not None:
                last_command = np.asarray(ros.last_state, dtype=np.float32)
                library_anchor = last_command.copy()
        else:
            print("[phase_teleop] DRY RUN: no ROS command publisher", flush=True)
            last_command = G20_OPEN_POSE.copy()
            library_anchor = G20_OPEN_POSE.copy()

        if args.freeze_thumb_on_start:
            hold_source = (
                np.asarray(ros.last_state, dtype=np.float32)
                if ros is not None and ros.last_state is not None
                else (
                    last_command.copy()
                    if last_command is not None
                    else None
                )
            )
            if hold_source is None:
                print(
                    "[phase_teleop] cannot freeze thumb on start: "
                    "no current DexHand pose",
                    file=sys.stderr,
                )
                return 2
            measured_thumb = np.rint(
                np.asarray(hold_source, dtype=np.float32)[list(THUMB_IDX)]
            ).astype(int).tolist()
            if args.startup_thumb_pose is not None:
                thumb_hold_pose = absolute_thumb_pose(
                    hold_source,
                    args.startup_thumb_pose,
                )
                target_thumb = np.rint(
                    thumb_hold_pose[list(THUMB_IDX)]
                ).astype(int).tolist()
                print(
                    "[phase_teleop] STARTUP THUMB ABSOLUTE q0/q5/q10/q15 "
                    f"measured={measured_thumb} target={target_thumb}; "
                    "startup reset will move then hold this target",
                    flush=True,
                )
            elif args.startup_thumb_offsets is None:
                thumb_hold_pose = np.asarray(hold_source, dtype=np.float32).copy()
                thumb_hold_pose[list(RESERVED_IDX)] = 255.0
                print(
                    "[phase_teleop] THUMB FROZEN ON START at q0/q5/q10/q15="
                    f"{measured_thumb}",
                    flush=True,
                )
            else:
                thumb_hold_pose = offset_thumb_pose(
                    hold_source,
                    args.startup_thumb_offsets,
                )
                target_thumb = np.rint(
                    thumb_hold_pose[list(THUMB_IDX)]
                ).astype(int).tolist()
                print(
                    "[phase_teleop] STARTUP THUMB MOVE q0/q5/q10/q15 "
                    f"measured={measured_thumb} offsets="
                    f"{list(args.startup_thumb_offsets)} target={target_thumb}; "
                    "startup reset will move then hold this target",
                    flush=True,
                )

        import cv2

        hand_source = MediaPipeHandSource(
            camera_index=args.camera_index,
            min_detection_confidence=args.min_detection_confidence,
            min_tracking_confidence=args.min_tracking_confidence,
            fps=args.rate,
        )
        pipeline = HandPipeline(
            hand_source, force_side=args.side, min_score=args.min_hand_score
        )
        action_map = "  ".join(
            f"{primitive.id}={primitive.name}"
            for primitive in library.primitives.values()
        )
        print(f"[phase_teleop] actions: {action_map}", flush=True)
        print(f"[phase_teleop] feature profile: {library.feature_profile}", flush=True)
        if args.control_mode == "hybrid-fingers":
            print(
                "[phase_teleop] hybrid q1..q4/q16..q19; "
                "thumb q0/q5/q10/q15 and spread q6..q9 are library-only",
                flush=True,
            )
            print(
                f"[phase_teleop] hybrid blend base={args.finger_base_blend:.2f} "
                f"limit={args.finger_base_residual_limit:.0f}; "
                f"tip={args.finger_tip_blend:.2f} "
                f"limit={args.finger_tip_residual_limit:.0f}",
                flush=True,
            )
            print(
                "[phase_teleop] unmatched fallback="
                f"{'direct MediaPipe four-finger flexion' if args.hybrid_unlocked_fingers else 'hold'}; "
                "thumb/spread hold their last library anchor",
                flush=True,
            )
        if args.tracking_mode == "live-pose":
            selected = "AUTO" if matcher.selected_id is None else str(matcher.selected_id)
            print(
                f"[phase_teleop] live selection={selected}; "
                "1-9 plays a library action without MediaPipe, 0=AUTO",
                flush=True,
            )
            print(
                "[phase_teleop] T: during a manual action toggle action-thumb/"
                "MediaPipe-fingers; otherwise toggle FULL MEDIAPIPE/AUTO",
                flush=True,
            )
            if a23_spread_router.enabled:
                print(
                    "[phase_teleop] A23 finger-spread router: "
                    f"threshold={a23_spread_router.threshold:.3f} "
                    f"hysteresis={a23_spread_router.hysteresis:.3f}; "
                    "together=ONLY A2, open=ALL ACTIONS",
                    flush=True,
                )
            if a3_contact_assist is not None and a3_contact_assist.enabled:
                if a3_contact_assist.roundtrip:
                    assist_note = (
                        "round-trip acquire at "
                        f"phase>={args.a3_contact_activate_phase:.0%} for "
                        f"{args.a3_contact_confirm_frames} frames; "
                        f"contact={a3_contact_assist.primitive.phase_contact_fraction:.0%}; "
                        "hold 100% until 0=AUTO"
                    )
                else:
                    assist_note = (
                        "bidirectional acquire at "
                        f"phase>={args.a3_contact_activate_phase:.0%} for "
                        f"{args.a3_contact_confirm_frames} frames; "
                        f"release<={args.a3_contact_release_phase:.0%}"
                    )
                print(f"[phase_teleop] A3 contact assist: {assist_note}", flush=True)
        if args.a4_thumb_tip_gate:
            print(
                "[phase_teleop] A4 ordered gate: left-align q0/q5/q10 "
                f"(error<={args.a4_left_align_tolerance:.0f} for "
                f"{args.a4_left_align_confirm_frames} frames), then close q15 "
                f"(error<={args.a4_thumb_tip_tolerance:.0f} for "
                f"{args.a4_thumb_tip_confirm_frames} frames), then turn right",
                flush=True,
            )
        reset_note = (
            f"reset-{episode_start_label} then arm"
            if args.reset_before_arm
            else "arm"
        )
        stop_note = (
            f"; armed SPACE stops the episode then resets {episode_start_label}"
            if args.reset_after_disarm
            else ""
        )
        print(
            f"[phase_teleop] starts DISARMED; focus window and press SPACE to "
            f"{reset_note}{stop_note}",
            flush=True,
        )
        if args.reset_on_start:
            resetting = True
            arm_after_reset = False
            reset_started = time.monotonic()
            reset_confirmed = 0
            status = (
                f"STARTUP RESET {episode_start_label.upper()}; "
                "ACT recorder remains stopped"
            )
            if ros is not None:
                ros.publish_session_active(False)
            print(f"[phase_teleop] {status}", flush=True)
        while True:
            if ros is not None:
                rclpy.spin_once(ros.node, timeout_sec=0.0)
            detection = hand_source.read()
            processed = pipeline.process(detection)
            fresh = processed is not None and processed.detected and not processed.held
            if fresh and a23_spread_router.enabled:
                # Keep the A2/A3 gate observable before arming and during
                # manual/full-MediaPipe modes, not only while AUTO is active.
                a23_spread_router.update(processed.landmarks)

            target = None
            active_primitive_id: Optional[int] = None
            manual_active = False
            if resetting:
                # Reset is human-triggered by SPACE.  The ACT trigger remains
                # false throughout; reset can either precede arming or follow
                # the end of an episode.
                lost_frames = 0
                last_result = None
                target = episode_start_pose
                status = (
                    f"RESET {episode_start_label.upper()}: "
                    "waiting for measured state"
                )
            elif armed and manual_frames is not None:
                # Normal playback is camera-independent. T can retain only its
                # thumb channels and hand every four-finger channel to MediaPipe.
                last_result = None
                assert manual_primitive_id is not None
                primitive = library.primitives[manual_primitive_id]
                primitive_name = (
                    f"thumb_action_{args.thumb_roundtrip_source_action}_roundtrip"
                    if manual_primitive_id == args.thumb_roundtrip_key
                    else primitive.name
                )
                active_primitive_id = manual_primitive_id
                manual_active = True
                frame_index = min(manual_frame, len(manual_frames) - 1)
                action_target = np.asarray(
                    manual_frames[frame_index], dtype=np.float32
                )
                if manual_thumb_hybrid:
                    if fresh:
                        lost_frames = 0
                        retargeted = retarget_fn(
                            processed.landmarks, side=args.side
                        )
                        finger_target = full_mediapipe_g20_target(
                            retargeted["joint_rad"]
                        )
                        hand_note = "live"
                    else:
                        lost_frames += 1
                        finger_target = (
                            last_command.copy()
                            if last_command is not None
                            else G20_OPEN_POSE.copy()
                        )
                        hand_note = "held (hand lost)"
                    target = action_thumb_mediapipe_fingers_target(
                        action_target, finger_target
                    )
                else:
                    lost_frames = 0
                    target = action_target
                library_anchor = np.asarray(target, dtype=np.float32).copy()
                if manual_thumb_hybrid:
                    stage = (
                        f"frame={frame_index + 1}/{len(manual_frames)}"
                        if manual_frame < len(manual_frames) - 1
                        else "HOLD endpoint"
                    )
                    status = (
                        f"MANUAL THUMB {manual_primitive_id}:{primitive_name} "
                        f"{stage}; MP four fingers={hand_note}"
                    )
                elif manual_frame < len(manual_frames) - 1:
                    status = (
                        f"MANUAL {manual_primitive_id}:{primitive_name} "
                        f"frame={frame_index + 1}/{len(manual_frames)}; MediaPipe ignored"
                    )
                else:
                    status = (
                        f"MANUAL HOLD {manual_primitive_id}:{primitive_name}; "
                        "press same key to replay or 0 for AUTO"
                    )
                if (
                    manual_primitive_id == args.thumb_roundtrip_key
                    and thumb_hold_pose is not None
                ):
                    status += (
                        f"; F hold suspended until action "
                        f"{args.thumb_roundtrip_key} is exited"
                    )
            elif fresh:
                lost_frames = 0
                if armed:
                    retargeted = retarget_fn(processed.landmarks, side=args.side)
                    if full_teleop:
                        # FULL MP owns every active channel. The matcher and A4
                        # gate are reset when T enters this mode.
                        last_result = None
                        target = full_mediapipe_g20_target(
                            retargeted["joint_rad"]
                        )
                        library_anchor = target.copy()
                        status = "FULL MEDIAPIPE TELEOP: action library bypassed"
                    else:
                        live_feature = library.feature(processed.landmarks)
                        a3_may_acquire = (
                            action3_assist_may_acquire(last_result)
                            and a23_spread_router.allows_a3_assist
                        )
                        if a3_contact_assist is not None and a3_may_acquire:
                            assisted_result = a3_contact_assist.update(live_feature)
                        else:
                            assisted_result = None
                            if a3_contact_assist is not None:
                                a3_contact_assist.reset()
                        if assisted_result is not None:
                            matcher.reset()
                            last_result = assisted_result
                        else:
                            if args.tracking_mode == "live-pose":
                                last_result = matcher.update(
                                    live_feature,
                                    excluded_ids=a23_spread_router.excluded_ids(
                                        library.primitives.keys()
                                    ),
                                )
                            else:
                                last_result = matcher.update(live_feature)
                        mediapipe_target = None
                        if args.control_mode == "hybrid-fingers":
                            mediapipe_target = nonthumb_radians_to_g20_target(
                                retargeted["joint_rad"],
                                base_gain=args.mediapipe_base_gain,
                                base_gains=args.mediapipe_base_gains,
                                tip_gain=args.mediapipe_tip_gain,
                                tip_gains=args.mediapipe_tip_gains,
                            )

                        if last_result.locked and last_result.primitive_id is not None:
                            primitive = library.primitives[last_result.primitive_id]
                            active_primitive_id = last_result.primitive_id
                            library_target = trajectory_target(
                                primitive.trajectory, last_result.phase
                            )
                            library_anchor = library_target.copy()
                            target = library_target
                            if (
                                args.control_mode == "hybrid-fingers"
                                and not a23_spread_router.freezes_four_fingers
                            ):
                                assert mediapipe_target is not None
                                target = hybrid_finger_target(
                                    library_target,
                                    mediapipe_target,
                                    base_blend=args.finger_base_blend,
                                    tip_blend=args.finger_tip_blend,
                                    base_residual_limit=args.finger_base_residual_limit,
                                    tip_residual_limit=args.finger_tip_residual_limit,
                                )
                            status = (
                                f"{last_result.primitive_id}:{last_result.name} "
                                f"phase={last_result.phase:.0%} d={last_result.distance:.4f}"
                            )
                        elif (
                            args.control_mode == "hybrid-fingers"
                            and args.hybrid_unlocked_fingers
                            and not a23_spread_router.freezes_four_fingers
                        ):
                            assert mediapipe_target is not None
                            if library_anchor is None:
                                library_anchor = (
                                    last_command.copy()
                                    if last_command is not None
                                    else G20_OPEN_POSE.copy()
                                )
                            target = mediapipe_finger_fallback_target(
                                library_anchor, mediapipe_target
                            )
                            candidate = (
                                "none"
                                if last_result.primitive_id is None
                                else f"{last_result.primitive_id}:{last_result.name}"
                            )
                            status = (
                                f"MP-FINGERS candidate={candidate}; "
                                "thumb/spread=library-hold"
                            )
                        else:
                            candidate = (
                                "none"
                                if last_result.primitive_id is None
                                else f"{last_result.primitive_id}:{last_result.name}"
                            )
                            status = (
                                f"SEARCHING candidate={candidate} "
                                f"phase={last_result.phase:.0%} "
                                f"d={last_result.distance:.4f}"
                            )
                        if a23_spread_router.enabled:
                            status += (
                                f" fingers={a23_spread_router.label} "
                                f"spread={a23_spread_router.score:.3f}"
                            )

            else:
                lost_frames += 1
                if armed and lost_frames >= max(1, args.hand_lost_frames):
                    armed = False
                    thumb_hold_pose = None
                    a4_thumb_tip_gate.reset()
                    matcher.reset()
                    a23_spread_router.reset()
                    if a3_contact_assist is not None:
                        a3_contact_assist.reset()
                    last_result = None
                    status = "hand lost; DISARMED and holding last pose"
                    if ros is not None:
                        ros.publish_session_active(False)
                    print(f"[phase_teleop] {status}", flush=True)

            if active_primitive_id != A4_PRIMITIVE_ID and target is None:
                a4_thumb_tip_gate.reset()

            if (armed or resetting) and target is not None:
                previous = last_command if last_command is not None else G20_OPEN_POSE
                observed = (
                    ros.last_state
                    if ros is not None and ros.last_state is not None
                    else None
                )
                close_pose = (
                    library.primitives[A4_PRIMITIVE_ID].trajectory[0]
                    if A4_PRIMITIVE_ID in library.primitives
                    else target
                )
                thumb_roundtrip_active = (
                    manual_active
                    and manual_primitive_id == args.thumb_roundtrip_key
                )
                if thumb_hold_pose is not None and not thumb_roundtrip_active:
                    # F supersedes normal/A4/reset thumb targets. The explicit
                    # key-6 thumb roundtrip temporarily owns the same channels,
                    # while the F latch remains available after key 6 is exited.
                    a4_thumb_tip_gate.reset()
                    target = frozen_thumb_target(target, thumb_hold_pose)
                    a4_waiting = False
                    a4_released_now = False
                    status = (
                        f"{status}; THUMB FROZEN "
                        f"q0/q5/q10/q15="
                        f"{np.rint(thumb_hold_pose[list(THUMB_IDX)]).astype(int).tolist()}"
                    )
                else:
                    target, a4_waiting, a4_released_now = a4_thumb_tip_gate.apply(
                        target,
                        primitive_id=active_primitive_id,
                        close_pose=close_pose,
                        observed=observed,
                        previous=previous,
                    )
                if a4_waiting:
                    if a4_thumb_tip_gate.last_applied_stage == "left":
                        status = (
                            "A4 LEFT ALIGN: q15 held; moving q0/q5/q10 left "
                            f"error={a4_thumb_tip_gate.last_error:.0f}/"
                            f"{args.a4_left_align_tolerance:.0f} "
                            f"settle={a4_thumb_tip_gate.confirmed}/"
                            f"{args.a4_left_align_confirm_frames}"
                        )
                    else:
                        status = (
                            "A4 TIP GATE: left aligned; closing q15 "
                            f"error={a4_thumb_tip_gate.last_error:.0f}/"
                            f"{args.a4_thumb_tip_tolerance:.0f} "
                            f"settle={a4_thumb_tip_gate.confirmed}/"
                            f"{args.a4_thumb_tip_confirm_frames}"
                        )
                elif a4_released_now:
                    status = "A4 TIP CLOSED: right rotation released"
                    if manual_active and manual_primitive_id == A4_PRIMITIVE_ID:
                        rebase_pose = (
                            np.asarray(observed, dtype=np.float32)
                            if observed is not None
                            else np.asarray(previous, dtype=np.float32)
                        )
                        primitive = library.primitives[A4_PRIMITIVE_ID]
                        source_trajectory = (
                            manual_source_trajectory
                            if manual_source_trajectory is not None
                            else primitive.trajectory
                        )
                        manual_frames = playback_trajectory(
                            source_trajectory,
                            start_pose=rebase_pose,
                            max_step=args.max_range_step,
                            blend_frames=args.manual_blend_frames,
                        )
                        manual_frame = 0
                        target = manual_frames[0]
                if manual_active and not a4_waiting and manual_frames is not None:
                    if manual_frame < len(manual_frames) - 1:
                        manual_frame += 1
                if active_primitive_id == A4_PRIMITIVE_ID:
                    library_anchor = np.asarray(target, dtype=np.float32).copy()
                command = state_guarded_command(
                    target,
                    previous,
                    observed,
                    max_step=args.max_range_step,
                    max_state_lead=args.max_state_lead,
                )
                last_command = np.asarray(command, dtype=np.float32)
                if ros is not None:
                    ros.publish_pose(command)
                if resetting:
                    feedback = (
                        np.asarray(ros.last_state, dtype=np.float32)
                        if ros is not None and ros.last_state is not None
                        else last_command
                    )
                    reset_target = (
                        frozen_thumb_target(episode_start_pose, thumb_hold_pose)
                        if thumb_hold_pose is not None
                        else episode_start_pose
                    )
                    error = active_pose_error(feedback, reset_target)
                    if error <= args.reset_tolerance:
                        reset_confirmed += 1
                    else:
                        reset_confirmed = 0
                    status = (
                        f"RESET {episode_start_label.upper()} "
                        f"error={error:.0f}/{args.reset_tolerance:.0f} "
                        f"settle={reset_confirmed}/{args.reset_confirm_frames}"
                        + (
                            " THUMB=FROZEN"
                            if thumb_hold_pose is not None
                            else ""
                        )
                    )
                    if reset_confirmed >= args.reset_confirm_frames:
                        resetting = False
                        reset_confirmed = 0
                        matcher.reset()
                        a23_spread_router.reset()
                        if a3_contact_assist is not None:
                            a3_contact_assist.reset()
                        library_anchor = feedback.copy()
                        completed_pose_label = (
                            f"{episode_start_label} with thumb frozen"
                            if thumb_hold_pose is not None
                            else episode_start_label
                        )
                        armed, status = reset_completion(
                            arm_after_reset, completed_pose_label
                        )
                        if armed:
                            if ros is not None:
                                ros.publish_session_active(True)
                        arm_after_reset = False
                        print(f"[phase_teleop] {status}", flush=True)
                    elif time.monotonic() - reset_started >= args.reset_timeout:
                        was_arming = arm_after_reset
                        resetting = False
                        arm_after_reset = False
                        reset_confirmed = 0
                        if ros is not None:
                            ros.publish_session_active(False)
                        status = (
                            f"RESET TIMEOUT error={error:.0f}; DISARMED, "
                            + (
                                "ACT episode not started"
                                if was_arming
                                else "completed episode remains saved"
                            )
                        )
                        print(f"[phase_teleop] {status}", flush=True)
                now = time.monotonic()
                if now - last_log >= 0.25:
                    last_log = now
                    print(
                        f"[phase_teleop] "
                        f"{'PUBLISH' if ros is not None else 'WOULD_CMD'} "
                        f"{status} cmd={command}",
                        flush=True,
                    )

            if (
                (armed or resetting)
                and ros is not None
                and time.monotonic() - state_clock["updated"] > args.state_stale_seconds
            ):
                armed = False
                resetting = False
                thumb_hold_pose = None
                arm_after_reset = False
                reset_confirmed = 0
                matcher.reset()
                a23_spread_router.reset()
                if a3_contact_assist is not None:
                    a3_contact_assist.reset()
                last_result = None
                ros.publish_session_active(False)
                status = "joint state stale; DISARMED and holding last pose"
                print(f"[phase_teleop] {status}", flush=True)

            frame = draw_hand_overlay(
                hand_source.last_frame_bgr,
                getattr(hand_source, "last_landmarks_raw_px", None),
                fresh=fresh,
            )
            _overlay(
                frame,
                armed=armed,
                fresh=fresh,
                status=status,
                locked=bool(last_result is not None and last_result.locked),
                manual=manual_frames is not None,
                full_teleop=full_teleop,
                thumb_frozen=thumb_hold_pose is not None,
                a23_enabled=a23_spread_router.enabled,
                a23_label=a23_spread_router.label,
                a23_score=a23_spread_router.score,
                a23_threshold=a23_spread_router.threshold,
                resetting=resetting,
                reset_before_arm=args.reset_before_arm,
                reset_after_disarm=args.reset_after_disarm,
                phase=0.0 if last_result is None else last_result.phase,
                minimal=args.minimal_overlay,
            )
            cv2.imshow("action library online phase teleop", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if is_delete_last_episode_key(key):
                a4_thumb_tip_gate.reset()
                manual_primitive_id = None
                manual_frames = None
                manual_source_trajectory = None
                manual_frame = 0
                manual_thumb_hybrid = False
                matcher.reset()
                a23_spread_router.reset()
                if a3_contact_assist is not None:
                    a3_contact_assist.reset()
                last_result = None
                if ros is not None and not armed and not resetting:
                    ros.publish_delete_last_episode()
                    status = (
                        "delete-last requested; manual/matcher reset; "
                        + (
                            "thumb remains frozen"
                            if thumb_hold_pose is not None
                            else "thumb follows current mode"
                        )
                    )
                elif armed or resetting:
                    status = (
                        "D: stop with SPACE and wait for reset before deleting "
                        "latest episode"
                    )
                else:
                    status = "DRY RUN: manual/matcher reset; no recorder request"
                print(f"[phase_teleop] {status}", flush=True)
            if is_manual_reset_key(key):
                if resetting:
                    status = "R: reset already in progress"
                elif armed:
                    status = "R: stop and save the active episode with SPACE first"
                else:
                    a4_thumb_tip_gate.reset()
                    matcher.reset()
                    a23_spread_router.reset()
                    if a3_contact_assist is not None:
                        a3_contact_assist.reset()
                    last_result = None
                    manual_primitive_id = None
                    manual_frames = None
                    manual_source_trajectory = None
                    manual_frame = 0
                    manual_thumb_hybrid = False
                    resetting = True
                    arm_after_reset = False
                    reset_started = time.monotonic()
                    reset_confirmed = 0
                    if ros is not None:
                        ros.publish_session_active(False)
                    status = (
                        f"R RESET {episode_start_label.upper()} started; "
                        "ACT recorder remains stopped"
                        + (
                            "; thumb remains frozen"
                            if thumb_hold_pose is not None
                            else ""
                        )
                    )
                print(f"[phase_teleop] {status}", flush=True)
            if args.tracking_mode == "live-pose" and is_full_teleop_toggle_key(key):
                if manual_frames is not None:
                    if manual_primitive_id == args.thumb_roundtrip_key:
                        manual_thumb_hybrid = False
                        full_teleop = False
                        status = (
                            f"T ignored for action {args.thumb_roundtrip_key}: "
                            "thumb follows action 2 roundtrip; four fingers stay fixed"
                        )
                    else:
                        manual_thumb_hybrid = not manual_thumb_hybrid
                        full_teleop = False
                        status = (
                            "MANUAL THUMB + MEDIAPIPE FOUR FINGERS"
                            if manual_thumb_hybrid
                            else "MANUAL FULL ACTION: MediaPipe ignored"
                        )
                else:
                    full_teleop = not full_teleop
                    manual_thumb_hybrid = False
                    a4_thumb_tip_gate.reset()
                    matcher.reset()
                    a23_spread_router.reset()
                    if a3_contact_assist is not None:
                        a3_contact_assist.reset()
                    last_result = None
                    status = (
                        "FULL MEDIAPIPE TELEOP: all active joints follow MediaPipe"
                        if full_teleop
                        else "AUTO hybrid MediaPipe/action-library tracking"
                    )
                print(f"[phase_teleop] {status}", flush=True)
            if is_thumb_freeze_toggle_key(key):
                if resetting:
                    status = (
                        "F: wait for reset to finish before changing thumb hold; "
                        "current hold remains active"
                    )
                elif thumb_hold_pose is not None:
                    thumb_hold_pose = None
                    status = "THUMB RELEASED: current teleop mode controls thumb again"
                else:
                    hold_source = (
                        np.asarray(ros.last_state, dtype=np.float32)
                        if ros is not None and ros.last_state is not None
                        else (
                            last_command.copy()
                            if last_command is not None
                            else None
                        )
                    )
                    if hold_source is None:
                        status = "F: no current DexHand pose available"
                    else:
                        thumb_hold_pose = np.asarray(
                            hold_source, dtype=np.float32
                        ).copy()
                        thumb_hold_pose[list(RESERVED_IDX)] = 255.0
                        a4_thumb_tip_gate.reset()
                        status = (
                            "THUMB FROZEN at q0/q5/q10/q15="
                            f"{np.rint(thumb_hold_pose[list(THUMB_IDX)]).astype(int).tolist()}; "
                            "press F again to release"
                        )
                print(f"[phase_teleop] {status}", flush=True)
            if args.tracking_mode == "live-pose" and key == ord("0"):
                full_teleop = False
                manual_thumb_hybrid = False
                a4_thumb_tip_gate.reset()
                manual_primitive_id = None
                manual_frames = None
                manual_source_trajectory = None
                manual_frame = 0
                matcher.select(None)
                a23_spread_router.reset()
                if a3_contact_assist is not None:
                    a3_contact_assist.reset()
                last_result = None
                status = "AUTO hybrid MediaPipe/action-library tracking"
                print(f"[phase_teleop] {status}", flush=True)
            if key in (ord("c"), ord("C")):
                if not armed or resetting:
                    status = "C: arm first and wait until reset is complete"
                elif manual_frames is not None:
                    status = "C: already completing/holding a manual trajectory"
                elif (
                    last_result is None
                    or last_result.primitive_id is None
                ):
                    status = "C: no locked/candidate action to complete"
                else:
                    requested = last_result.primitive_id
                    start_phase = float(last_result.phase)
                    source_kind = "locked" if last_result.locked else "candidate"
                    primitive = library.primitives[requested]
                    start_pose = (
                        np.asarray(ros.last_state, dtype=np.float32)
                        if ros is not None and ros.last_state is not None
                        else (
                            last_command.copy()
                            if last_command is not None
                            else G20_OPEN_POSE.copy()
                        )
                    )
                    remaining = trajectory_suffix(
                        primitive.trajectory, start_phase
                    )
                    manual_source_trajectory = remaining.copy()
                    manual_frames = playback_trajectory(
                        remaining,
                        start_pose=start_pose,
                        max_step=args.max_range_step,
                        blend_frames=args.manual_blend_frames,
                    )
                    manual_primitive_id = requested
                    manual_frame = 0
                    manual_thumb_hybrid = False
                    full_teleop = False
                    if requested != A4_PRIMITIVE_ID:
                        a4_thumb_tip_gate.reset()
                    matcher.reset()
                    a23_spread_router.reset()
                    if a3_contact_assist is not None:
                        a3_contact_assist.reset()
                    last_result = None
                    status = (
                        f"CONTINUE {requested}:{primitive.name} from "
                        f"{source_kind} phase={start_phase:.0%} to 100%; "
                        "MediaPipe ignored"
                    )
                print(f"[phase_teleop] {status}", flush=True)
            if args.tracking_mode == "live-pose" and ord("1") <= key <= ord("9"):
                requested = key - ord("0")
                if requested in library.primitives:
                    a4_thumb_tip_gate.reset()
                    primitive = library.primitives[requested]
                    start_pose = (
                        np.asarray(ros.last_state, dtype=np.float32)
                        if ros is not None and ros.last_state is not None
                        else (
                            last_command.copy()
                            if last_command is not None
                            else G20_OPEN_POSE.copy()
                        )
                    )
                    if requested == args.thumb_roundtrip_key:
                        source_primitive = library.primitives[
                            args.thumb_roundtrip_source_action
                        ]
                        remaining, nearest_frame, nearest_error = (
                            thumb_roundtrip_trajectory(
                                source_primitive.trajectory, start_pose
                            )
                        )
                        selection = (
                            f"thumb nearest A{source_primitive.id} frame "
                            f"{nearest_frame + 1}/"
                            f"{len(source_primitive.trajectory)} "
                            "-> endpoint -> frame 1; four fingers fixed"
                        )
                        if thumb_hold_pose is not None:
                            selection += "; F hold temporarily bypassed"
                        display_name = (
                            f"thumb_action_{source_primitive.id}_roundtrip"
                        )
                    else:
                        force_from_start = manual_action_starts_from_first_frame(
                            primitive, args.manual_action_from_start
                        )
                        remaining, nearest_frame, nearest_error = (
                            selected_manual_trajectory(
                                primitive.trajectory,
                                start_pose,
                                force_from_start=force_from_start,
                            )
                        )
                        selection = (
                            "FULL FROM FRAME 1"
                            if force_from_start
                            else f"from nearest frame {nearest_frame + 1}"
                        )
                        display_name = primitive.name
                    manual_source_trajectory = remaining.copy()
                    manual_frames = playback_trajectory(
                        remaining,
                        start_pose=start_pose,
                        max_step=args.max_range_step,
                        blend_frames=args.manual_blend_frames,
                    )
                    manual_primitive_id = requested
                    manual_frame = 0
                    manual_thumb_hybrid = False
                    full_teleop = False
                    matcher.reset()
                    a23_spread_router.reset()
                    if a3_contact_assist is not None:
                        a3_contact_assist.reset()
                    last_result = None
                    status = (
                        f"MANUAL {'START' if armed else 'QUEUED'} "
                        f"{requested}:{display_name} {selection} "
                        f"frames={len(remaining)} rms={nearest_error:.1f}; "
                        "MediaPipe ignored"
                    )
                    print(f"[phase_teleop] {status}", flush=True)
            if key == ord(" "):
                a4_thumb_tip_gate.reset()
                matcher.reset()
                a23_spread_router.reset()
                if a3_contact_assist is not None:
                    a3_contact_assist.reset()
                last_result = None
                if resetting:
                    resetting = False
                    arm_after_reset = False
                    reset_confirmed = 0
                    if ros is not None:
                        ros.publish_session_active(False)
                    status = "RESET CANCELLED; DISARMED and no ACT episode started"
                elif armed:
                    armed = False
                    if ros is not None:
                        ros.publish_session_active(False)
                    if args.reset_after_disarm:
                        resetting = True
                        arm_after_reset = False
                        reset_started = time.monotonic()
                        reset_confirmed = 0
                        status = (
                            "ACT episode stopped; RESET "
                            f"{episode_start_label.upper()} started while DISARMED"
                        )
                    else:
                        status = "DISARMED: holding last pose; ACT episode stopped"
                else:
                    if ros is not None and ros.last_state is not None:
                        last_command = np.asarray(ros.last_state, dtype=np.float32)
                    if last_command is not None:
                        library_anchor = last_command.copy()
                    if args.reset_before_arm:
                        resetting = True
                        arm_after_reset = True
                        reset_started = time.monotonic()
                        reset_confirmed = 0
                        if ros is not None:
                            ros.publish_session_active(False)
                        status = (
                            f"RESET {episode_start_label.upper()} started; "
                            "ACT recorder remains stopped"
                        )
                    else:
                        armed = True
                        arm_after_reset = False
                        if ros is not None:
                            ros.publish_session_active(True)
                        status = f"ARMED: {args.tracking_mode} tracking"
                print(f"[phase_teleop] {status}", flush=True)
        return 0
    except KeyboardInterrupt:
        return 130
    finally:
        if ros is not None:
            if rclpy is not None and rclpy.ok():
                try:
                    ros.publish_session_active(False)
                except Exception as exc:
                    print(f"[phase_teleop] cleanup hold warning: {exc}", file=sys.stderr)
            ros.close()
        if hand_source is not None:
            hand_source.close()
        try:
            import cv2

            cv2.destroyAllWindows()
        except ImportError:
            pass
        if state_watch is not None:
            del state_watch
        if rclpy is not None and rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
