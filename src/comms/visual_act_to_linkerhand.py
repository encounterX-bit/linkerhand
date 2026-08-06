#!/usr/bin/env python3
"""Run the camera-conditioned G20 ACT policy with conservative hardware gates.

The process always starts DISARMED and continues to show predictions.  Hardware
publishing requires all three of:

1. ``HW_ENABLE_TOKEN=1`` set by the human operator;
2. ``--enable-motion`` on the command line;
3. SPACE pressed in the camera window.

SPACE disarms again, R resets to the standard G20 open pose while disarmed,
and Q/ESC exits.  The default rate, step, current, speed, and active-time limits
are deliberately conservative for first bring-up. With
``--record-rated-attempts``, every stopped attempt must be rated in the camera
window with 0 (bad), 5 (partial/0.5), or 1 (good).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import get_policy_class, make_pre_post_processors

from src.comms.action_library import ActionLibrary


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT = (
    REPO_ROOT
    / "artifacts/g20_visual_act/training/checkpoints/010000/pretrained_model"
)
RESERVED_IDX = (11, 12, 13, 14)
JOINT_COUNT = 20
THUMB_ORIENTATION_IDX = (0, 5, 10)
THUMB_CONTROL_IDX = (0, 5, 10, 15)
ACTIVE_IDX = tuple(i for i in range(JOINT_COUNT) if i not in RESERVED_IDX)
A4_PRETURN_THUMB_ORIENTATION = (254.0, 0.0, 51.0)
MASS_KEYS = (
    "thumb_mass",
    "index_mass",
    "middle_mass",
    "ring_mass",
    "little_mass",
    "palm_mass",
)
FINGER_NAMES = ("thumb", "index", "middle", "ring", "little")
G20_OPEN_POSE = [
    255, 255, 255, 255, 255,
    255, 193, 148, 105, 42,
    245, 255, 255, 255, 255,
    255, 255, 255, 255, 255,
]


class ChunkBoundaryBlender:
    """Cross-fade only the first commands after an ACT chunk boundary."""

    def __init__(
        self,
        action_horizon: int,
        blend_frames: int,
        blend_indices: tuple[int, ...] | None = None,
    ) -> None:
        self.action_horizon = int(action_horizon)
        self.blend_frames = int(blend_frames)
        self.blend_indices = blend_indices
        self.reset()

    def reset(self) -> None:
        self.action_index = 0
        self.anchor: np.ndarray | None = None
        self.blend_index = 0

    def apply(
        self,
        raw: np.ndarray,
        previous_output: np.ndarray | None,
    ) -> tuple[np.ndarray, bool, float, bool]:
        raw = np.asarray(raw, dtype=np.float32)
        boundary_started = bool(
            self.blend_frames > 0
            and self.action_index > 0
            and self.action_index % self.action_horizon == 0
            and previous_output is not None
        )
        if boundary_started:
            self.anchor = np.asarray(previous_output, dtype=np.float32).copy()
            self.blend_index = 0

        active = self.anchor is not None
        weight = 1.0
        output = raw
        if active:
            weight = min(1.0, (self.blend_index + 1) / self.blend_frames)
            if self.blend_indices is None:
                output = (1.0 - weight) * self.anchor + weight * raw
            else:
                output = raw.copy()
                idx = list(self.blend_indices)
                output[idx] = (
                    (1.0 - weight) * self.anchor[idx]
                    + weight * raw[idx]
                )
            self.blend_index += 1
            if self.blend_index >= self.blend_frames:
                self.anchor = None

        self.action_index += 1
        return output.astype(np.float32), active, weight, boundary_started


class ActionLibraryIntervention:
    """Safely play a numbered action from its nearest measured-state frame.

    Playback holds the last library frame until the operator explicitly
    returns control to ACT. The main command path still applies keyboard thumb
    biases and measured-state step limiting after this controller.
    """

    def __init__(
        self,
        library: ActionLibrary,
        *,
        max_step: int,
        blend_frames: int,
    ) -> None:
        if max_step <= 0:
            raise ValueError("action intervention max_step must be positive")
        if blend_frames < 0:
            raise ValueError("action intervention blend_frames must be nonnegative")
        self.library = library
        self.max_step = int(max_step)
        self.blend_frames = int(blend_frames)
        self.stop()

    @property
    def active(self) -> bool:
        return self.action_id is not None and self.trajectory is not None

    @property
    def status(self) -> str | None:
        if not self.active:
            return None
        assert self.trajectory is not None and self.action_id is not None
        stage = "HOLD" if self.holding else "PLAY"
        return (
            f"A{self.action_id} {stage} "
            f"{min(self.frame_index + 1, len(self.trajectory))}/"
            f"{len(self.trajectory)}"
        )

    def stop(self) -> None:
        self.action_id: int | None = None
        self.trajectory: np.ndarray | None = None
        self.frame_index = 0
        self.holding = False
        self.nearest_source_frame = 0
        self.nearest_error = float("inf")

    def begin(
        self, action_id: int, observed_pose: list[int]
    ) -> tuple[int, float, int]:
        if action_id not in self.library.primitives:
            raise KeyError(action_id)
        observed = np.asarray(observed_pose, dtype=np.float32).reshape(-1)
        if observed.shape != (JOINT_COUNT,) or not np.all(np.isfinite(observed)):
            raise ValueError("observed pose must contain 20 finite values")
        source = np.asarray(
            self.library.primitives[action_id].trajectory, dtype=np.float32
        )
        differences = source[:, list(ACTIVE_IDX)] - observed[list(ACTIVE_IDX)]
        errors = np.sqrt(np.mean(np.square(differences), axis=1))
        nearest = int(np.argmin(errors))
        selected = source[nearest:].copy()
        selected[:, list(RESERVED_IDX)] = 255.0
        self.trajectory = self._safe_playback(selected, observed)
        self.action_id = int(action_id)
        self.frame_index = 0
        self.holding = False
        self.nearest_source_frame = nearest
        self.nearest_error = float(errors[nearest])
        return nearest, self.nearest_error, len(self.trajectory)

    def next_target(self) -> list[int]:
        if not self.active:
            raise RuntimeError("action-library intervention is not active")
        assert self.trajectory is not None
        index = min(self.frame_index, len(self.trajectory) - 1)
        target = clamp_pose(self.trajectory[index])
        if self.frame_index < len(self.trajectory) - 1:
            self.frame_index += 1
        else:
            self.holding = True
        return target

    def _safe_playback(
        self, selected: np.ndarray, observed: np.ndarray
    ) -> np.ndarray:
        active = list(ACTIVE_IDX)
        largest = float(np.max(np.abs(selected[0, active] - observed[active])))
        blend_count = max(
            self.blend_frames,
            int(np.ceil(largest / float(self.max_step))),
        )
        frames: list[np.ndarray] = []
        if blend_count:
            for alpha in np.linspace(1.0 / blend_count, 1.0, blend_count):
                frames.append((1.0 - alpha) * observed + alpha * selected[0])
        else:
            frames.append(selected[0].copy())
        previous = frames[-1]
        for target in selected[1:]:
            largest = float(np.max(np.abs(target[active] - previous[active])))
            count = max(1, int(np.ceil(largest / float(self.max_step))))
            for alpha in np.linspace(1.0 / count, 1.0, count):
                frames.append((1.0 - alpha) * previous + alpha * target)
            previous = target
        result = np.asarray(frames, dtype=np.float32)
        result[:, list(RESERVED_IDX)] = 255.0
        return result


def parse_g20_pose(value: str) -> tuple[float, ...]:
    """Parse one absolute 20-D SDK pose for a policy handoff."""
    try:
        pose = tuple(float(item.strip()) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "pose must contain 20 comma-separated numbers"
        ) from exc
    if len(pose) != JOINT_COUNT:
        raise argparse.ArgumentTypeError(
            f"pose must contain exactly {JOINT_COUNT} values"
        )
    if not np.all(np.isfinite(pose)) or any(value < 0 or value > 255 for value in pose):
        raise argparse.ArgumentTypeError("pose values must be finite and in [0, 255]")
    return pose


@dataclass(frozen=True)
class CubeMarkerPose:
    """Resolution-independent pose proxy for one visible tagged cube face."""

    center_x: float
    center_y: float
    width: float
    height: float
    marker_count: int
    face_id: int


def detect_cube_marker_pose(
    frame: np.ndarray, *, min_markers: int = 3
) -> CubeMarkerPose | None:
    """Return the dominant four-marker cube-face box in normalized pixels."""
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    corners, ids, _rejected = cv2.aruco.detectMarkers(frame, dictionary)
    if ids is None:
        return None
    flat_ids = ids.reshape(-1)
    face_counts: dict[int, int] = {}
    for marker_id in flat_ids:
        face_id = int(marker_id) // 4
        face_counts[face_id] = face_counts.get(face_id, 0) + 1
    face_id, marker_count = max(face_counts.items(), key=lambda item: item[1])
    if marker_count < min_markers:
        return None
    points = np.concatenate([
        np.asarray(marker_corners, dtype=np.float32).reshape(4, 2)
        for marker_corners, marker_id in zip(corners, flat_ids)
        if int(marker_id) // 4 == face_id
    ])
    low = points.min(axis=0)
    high = points.max(axis=0)
    frame_height, frame_width = frame.shape[:2]
    center = (low + high) * 0.5 / np.asarray([frame_width, frame_height])
    size = (high - low) / np.asarray([frame_width, frame_height])
    if np.any(size <= 1e-6):
        return None
    return CubeMarkerPose(
        center_x=float(center[0]),
        center_y=float(center[1]),
        width=float(size[0]),
        height=float(size[1]),
        marker_count=marker_count,
        face_id=face_id,
    )


def cube_ready_errors(
    live: CubeMarkerPose, reference: CubeMarkerPose
) -> tuple[float, float]:
    center_error = float(np.hypot(
        live.center_x - reference.center_x,
        live.center_y - reference.center_y,
    ))
    scale_error = float(max(
        abs(np.log(live.width / reference.width)),
        abs(np.log(live.height / reference.height)),
    ))
    return center_error, scale_error


def cube_pose_matches(
    live: CubeMarkerPose,
    reference: CubeMarkerPose,
    *,
    center_tolerance: float,
    scale_tolerance: float | None,
) -> tuple[bool, float, float]:
    center_error, scale_error = cube_ready_errors(live, reference)
    return (
        center_error <= center_tolerance
        and (scale_tolerance is None or scale_error <= scale_tolerance),
        center_error,
        scale_error,
    )


def load_cube_ready_profile(path: Path) -> CubeMarkerPose | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        pose = value["pose"] if isinstance(value, dict) and "pose" in value else value
        return CubeMarkerPose(**pose)
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        raise RuntimeError(f"invalid cube-ready profile: {path}")


def save_cube_ready_profile(path: Path, pose: CubeMarkerPose) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "linkerhand_cube_ready_pose_v1",
        "pose": asdict(pose),
        "note": "normalized ArUco bounding box; face_id is diagnostic only",
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def flatten_sum(value: Any) -> float:
    """Sum non-negative numeric values in a scalar or nested touch payload."""
    total = 0.0
    stack = [value]
    while stack:
        item = stack.pop()
        if isinstance(item, dict):
            stack.extend(item.values())
        elif isinstance(item, (list, tuple)):
            stack.extend(item)
        elif isinstance(item, (int, float)) and not isinstance(item, bool) and item >= 0:
            total += float(item)
    return total


def finger_mass_contact_count(
    contact_sample: dict[str, Any], threshold: float
) -> int:
    """Count fingers above a light pre-arm contact threshold."""
    masses = contact_sample.get("mass_values")
    if not contact_sample.get("touch_fresh") or not isinstance(masses, list):
        return 0
    return sum(float(value) >= threshold for value in masses[:5])


class TouchContactTracker:
    """Track hysteretic finger contacts and a stable final grasp condition."""

    def __init__(
        self,
        on_threshold: float,
        off_threshold: float,
        min_fingers: int,
        require_thumb: bool,
        hold_seconds: float,
    ) -> None:
        self.on_threshold = float(on_threshold)
        self.off_threshold = float(off_threshold)
        self.min_fingers = int(min_fingers)
        self.require_thumb = bool(require_thumb)
        self.hold_seconds = float(hold_seconds)
        self.contacts = [False] * 5
        self.palm_contact = False
        self.last_values: Optional[list[float]] = None
        self.last_update = 0.0
        self.qualifying_since: Optional[float] = None
        self.attempt_active = False
        self.peak_contact_count = 0
        self.max_finger_mass = [0.0] * 5

    def begin_attempt(self, now: float) -> None:
        self.attempt_active = True
        self.peak_contact_count = sum(self.contacts)
        self.max_finger_mass = (
            list(self.last_values[:5]) if self.last_values is not None else [0.0] * 5
        )
        self.qualifying_since = now if self._qualifies() else None

    def update(self, mass_values: list[float], now: float) -> None:
        if len(mass_values) < 6:
            return
        values = [max(0.0, float(value)) for value in mass_values[:6]]
        for index, value in enumerate(values[:5]):
            if self.contacts[index]:
                if value <= self.off_threshold:
                    self.contacts[index] = False
            elif value >= self.on_threshold:
                self.contacts[index] = True
        if self.palm_contact:
            if values[5] <= self.off_threshold:
                self.palm_contact = False
        elif values[5] >= self.on_threshold:
            self.palm_contact = True
        self.last_values = values
        self.last_update = now
        if self.attempt_active:
            self.peak_contact_count = max(self.peak_contact_count, sum(self.contacts))
            self.max_finger_mass = [
                max(old, new) for old, new in zip(self.max_finger_mass, values[:5])
            ]
            if self._qualifies():
                if self.qualifying_since is None:
                    self.qualifying_since = now
            else:
                self.qualifying_since = None

    def _qualifies(self) -> bool:
        enough_fingers = sum(self.contacts) >= self.min_fingers
        return enough_fingers and (self.contacts[0] or not self.require_thumb)

    def snapshot(self, now: float, stale_seconds: float) -> dict[str, Any]:
        age = None if self.last_values is None else max(0.0, now - self.last_update)
        fresh = age is not None and age <= stale_seconds
        held = (
            max(0.0, now - self.qualifying_since)
            if fresh and self.qualifying_since is not None and self._qualifies()
            else 0.0
        )
        stable = fresh and held >= self.hold_seconds
        return {
            "mass_keys": list(MASS_KEYS),
            "mass_values": list(self.last_values) if self.last_values is not None else None,
            "contacts": [int(value) for value in self.contacts]
            + [int(self.palm_contact)],
            "contact_finger_names": [
                name for name, active in zip(FINGER_NAMES, self.contacts) if active
            ],
            "finger_contact_count": sum(self.contacts),
            "thumb_contact": bool(self.contacts[0]),
            "continuous_contact_seconds": held,
            "touch_age_seconds": age,
            "touch_fresh": fresh,
            "success_gate_met": stable,
            "minimum_fingers": self.min_fingers,
            "thumb_required": self.require_thumb,
            "hold_seconds": self.hold_seconds,
            "peak_contact_count": self.peak_contact_count,
            "max_finger_mass": dict(zip(FINGER_NAMES, self.max_finger_mass)),
        }

    def end_attempt(self) -> None:
        self.attempt_active = False


def clamp_pose(values: np.ndarray) -> list[int]:
    pose = np.rint(np.asarray(values, dtype=np.float32).reshape(-1)[:JOINT_COUNT])
    if pose.size != JOINT_COUNT:
        raise RuntimeError(f"ACT returned {pose.size} values; expected {JOINT_COUNT}")
    pose = np.clip(pose, 0, 255).astype(np.int32)
    for idx in RESERVED_IDX:
        pose[idx] = 255
    return pose.tolist()


def limit_from_observed_state(target: list[int], state: list[int], step: int) -> list[int]:
    if step <= 0:
        return list(target)
    target_arr = np.asarray(target, dtype=np.int32)
    state_arr = np.asarray(state, dtype=np.int32)
    limited = state_arr + np.clip(target_arr - state_arr, -step, step)
    limited = np.clip(limited, 0, 255)
    for idx in RESERVED_IDX:
        limited[idx] = 255
    return limited.astype(np.int32).tolist()


def observed_hold_pose(state: list[int]) -> list[int]:
    """Make a one-shot hold command from fresh measured joint feedback."""
    pose = np.clip(
        np.rint(np.asarray(state, dtype=np.float32).reshape(-1)), 0, 255
    ).astype(np.int32)
    if pose.size != JOINT_COUNT:
        raise ValueError(f"observed state has {pose.size} values; expected {JOINT_COUNT}")
    for idx in RESERVED_IDX:
        pose[idx] = 255
    return pose.tolist()


@dataclass
class PolicyHandoffController:
    """Feedback-gated move and history warmup before a one-way policy switch."""

    target_pose: tuple[float, ...]
    tolerance: float
    confirm_frames: int
    warmup_seconds: float
    timeout_seconds: float
    phase: str = "idle"
    started_at: float = 0.0
    warmup_started_at: float | None = None
    confirmed: int = 0
    last_error: float = float("inf")

    @property
    def active(self) -> bool:
        return self.phase in ("move", "warmup")

    def begin(self, now: float) -> None:
        self.phase = "move"
        self.started_at = float(now)
        self.warmup_started_at = None
        self.confirmed = 0
        self.last_error = float("inf")

    def abort(self) -> None:
        self.phase = "idle"
        self.warmup_started_at = None
        self.confirmed = 0

    def target(self) -> list[int]:
        return clamp_pose(np.asarray(self.target_pose, dtype=np.float32))

    def command(self, observed: list[int], step: int) -> list[int]:
        return limit_from_observed_state(self.target(), observed, step)

    def update(self, observed: list[int], now: float) -> str:
        if not self.active:
            return self.phase
        if now - self.started_at > self.timeout_seconds:
            self.phase = "timeout"
            return self.phase
        target = np.asarray(self.target(), dtype=np.float32)
        state = np.asarray(observed, dtype=np.float32)
        active_indices = [index for index in range(JOINT_COUNT) if index not in RESERVED_IDX]
        self.last_error = float(
            np.max(np.abs(target[active_indices] - state[active_indices]))
        )
        if self.phase == "move":
            if self.last_error <= self.tolerance:
                self.confirmed += 1
            else:
                self.confirmed = 0
            if self.confirmed >= self.confirm_frames:
                self.phase = "warmup"
                self.warmup_started_at = float(now)
                return "warmup_started"
            return "moving"

        assert self.phase == "warmup"
        if self.last_error > self.tolerance:
            self.warmup_started_at = float(now)
            return "warmup_reset"
        assert self.warmup_started_at is not None
        if now - self.warmup_started_at >= self.warmup_seconds:
            self.phase = "complete"
            return self.phase
        return "warming"


def apply_thumb_final_push_offset(
    target: list[int], observed: list[int], offset: int
) -> tuple[list[int], bool]:
    """Add a small q15 extension only at the aligned end of the A4 push."""
    desired = np.asarray(target, dtype=np.int32).reshape(-1)
    feedback = np.asarray(observed, dtype=np.int32).reshape(-1)
    if desired.size != JOINT_COUNT or feedback.size != JOINT_COUNT:
        raise ValueError("thumb final-push target and state must contain 20 values")
    enabled = int(offset) != 0
    target_is_final_push = bool(
        desired[0] <= 125
        and desired[5] >= 120
        and desired[10] <= 45
        and desired[15] >= 48
    )
    hand_is_aligned = bool(
        feedback[0] <= 145
        and feedback[5] >= 110
        and feedback[10] <= 60
    )
    if not enabled or not target_is_final_push or not hand_is_aligned:
        return desired.tolist(), False
    output = desired.copy()
    output[15] = int(np.clip(output[15] + int(offset), 0, 255))
    return output.tolist(), True


def apply_thumb_joint_bias(
    target: list[int], joint_index: int, bias: int
) -> list[int]:
    """Apply a bounded additive bias to one G20 thumb command channel."""
    desired = np.asarray(target, dtype=np.int32).reshape(-1)
    if desired.size != JOINT_COUNT:
        raise ValueError("thumb-bias target must contain 20 values")
    if joint_index not in THUMB_CONTROL_IDX:
        raise ValueError(
            f"thumb-bias joint must be one of {THUMB_CONTROL_IDX}, got {joint_index}"
        )
    output = desired.copy()
    output[joint_index] = int(
        np.clip(output[joint_index] + int(bias), 0, 255)
    )
    return output.tolist()


def adjust_bounded_keyboard_bias(current: int, delta: int, limit: int) -> int:
    """Apply one keyboard step while keeping a symmetric bias limit."""
    if limit < 0:
        raise ValueError("keyboard bias limit must be non-negative")
    return int(np.clip(int(current) + int(delta), -int(limit), int(limit)))


def load_demo_endpoint_profile(path: Path) -> tuple[tuple[int, ...], np.ndarray]:
    """Load successful demonstration endpoint states used for auto-stop."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot load endpoint profile {path}: {exc}") from exc
    if data.get("schema") != "g20_demo_endpoint_profile_v1":
        raise RuntimeError(f"unsupported endpoint profile schema in {path}")
    active_indices = tuple(int(value) for value in data.get("active_indices", []))
    if (
        not active_indices
        or len(set(active_indices)) != len(active_indices)
        or any(index < 0 or index >= JOINT_COUNT for index in active_indices)
        or any(index in RESERVED_IDX for index in active_indices)
    ):
        raise RuntimeError(f"invalid active_indices in endpoint profile {path}")
    templates = data.get("templates")
    if not isinstance(templates, list) or not templates:
        raise RuntimeError(f"endpoint profile {path} contains no templates")
    positions: list[list[float]] = []
    for item in templates:
        position = item.get("position") if isinstance(item, dict) else None
        if not isinstance(position, list) or len(position) != JOINT_COUNT:
            raise RuntimeError(f"invalid endpoint template in {path}")
        values = [float(value) for value in position]
        if not np.all(np.isfinite(values)):
            raise RuntimeError(f"non-finite endpoint template in {path}")
        positions.append(values)
    return active_indices, np.asarray(positions, dtype=np.float32)


class DemoEndpointStopper:
    """Detect return to any successful demo endpoint after real task motion."""

    def __init__(
        self,
        templates: np.ndarray,
        active_indices: tuple[int, ...],
        *,
        tolerance: float,
        confirm_frames: int,
        min_active_seconds: float,
        departure_delta: float,
    ) -> None:
        values = np.asarray(templates, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != JOINT_COUNT or values.shape[0] == 0:
            raise ValueError("endpoint templates must have shape (N, 20)")
        self.templates = values
        self.active_indices = np.asarray(active_indices, dtype=np.int64)
        self.tolerance = float(tolerance)
        self.confirm_frames = int(confirm_frames)
        self.min_active_seconds = float(min_active_seconds)
        self.departure_delta = float(departure_delta)
        self.reset()

    def reset(self, start_state: list[int] | None = None, now: float = 0.0) -> None:
        self.start_state = (
            None
            if start_state is None
            else np.asarray(start_state, dtype=np.float32).reshape(-1)
        )
        self.started_at = float(now)
        self.departed = False
        self.confirmed = 0
        self.nearest_error = float("inf")

    def update(self, state: list[int], now: float) -> bool:
        observed = np.asarray(state, dtype=np.float32).reshape(-1)
        if observed.size != JOINT_COUNT:
            raise ValueError("endpoint stopper state must contain 20 values")
        if self.start_state is None:
            self.reset(state, now)
            return False
        if not self.departed:
            start_delta = float(
                np.max(
                    np.abs(
                        observed[self.active_indices]
                        - self.start_state[self.active_indices]
                    )
                )
            )
            self.departed = start_delta >= self.departure_delta
        errors = np.max(
            np.abs(
                self.templates[:, self.active_indices]
                - observed[None, self.active_indices]
            ),
            axis=1,
        )
        self.nearest_error = float(np.min(errors))
        eligible = (
            self.departed
            and float(now) - self.started_at >= self.min_active_seconds
            and self.nearest_error <= self.tolerance
        )
        self.confirmed = self.confirmed + 1 if eligible else 0
        return self.confirmed >= self.confirm_frames


class ThumbTipBeforeTurnGate:
    """Hold the recorded A4 pre-turn pose until the measured thumb tip closes.

    ACT predicts an absolute multi-joint target, so q0/q5/q10 can begin the
    right turn while physical q15 is still catching up. The demonstrations
    consistently wait at q0/q5/q10 ~= 254/0/51, close q15 to zero, and only
    then turn. This feedback gate restores that ordering at inference time.
    """

    def __init__(
        self,
        *,
        release_threshold: float,
        confirm_frames: int,
        activation_tolerance: float = 35.0,
        turn_delta: float = 3.0,
        turn_release_q0: float = 220.0,
        turn_release_q10: float = 10.0,
        turn_confirm_frames: int = 3,
        enabled: bool = True,
    ):
        self.release_threshold = float(release_threshold)
        self.confirm_frames = int(confirm_frames)
        self.activation_tolerance = float(activation_tolerance)
        self.turn_delta = float(turn_delta)
        self.turn_release_q0 = float(turn_release_q0)
        self.turn_release_q10 = float(turn_release_q10)
        self.turn_confirm_frames = int(turn_confirm_frames)
        self.enabled = bool(enabled)
        self.reset()

    def reset(self) -> None:
        self.active = False
        self.released = False
        self.confirmed = 0
        self.stage = "idle"
        self.last_tip_position = float("inf")
        self.last_q0 = float("inf")
        self.last_q10 = float("inf")

    def apply(
        self, target: list[int], observed: list[int]
    ) -> tuple[list[int], bool, bool]:
        desired = np.asarray(target, dtype=np.float32).reshape(-1)
        feedback = np.asarray(observed, dtype=np.float32).reshape(-1)
        if (
            desired.shape != (JOINT_COUNT,)
            or feedback.shape != (JOINT_COUNT,)
            or not np.all(np.isfinite(desired))
            or not np.all(np.isfinite(feedback))
        ):
            raise ValueError("thumb gate target and feedback must contain 20 values")
        if not self.enabled or self.released:
            return desired.astype(np.int32).tolist(), False, False

        preturn = np.asarray(A4_PRETURN_THUMB_ORIENTATION, dtype=np.float32)
        orientation_feedback = feedback[list(THUMB_ORIENTATION_IDX)]
        orientation_target = desired[list(THUMB_ORIENTATION_IDX)]
        near_preturn = bool(
            np.max(np.abs(orientation_feedback - preturn))
            <= self.activation_tolerance
        )
        turn_requested = bool(
            np.max(np.abs(orientation_target - preturn)) >= self.turn_delta
        )
        tip_not_closed = bool(feedback[15] > self.release_threshold)
        if not self.active and near_preturn and turn_requested and tip_not_closed:
            self.active = True
            self.stage = "tip"

        if not self.active:
            return desired.astype(np.int32).tolist(), False, False

        output = desired.copy()
        output[15] = 0.0
        output[list(RESERVED_IDX)] = 255.0

        if self.stage == "tip":
            output[list(THUMB_ORIENTATION_IDX)] = preturn
            self.last_tip_position = float(feedback[15])
            if self.last_tip_position <= self.release_threshold:
                self.confirmed += 1
            else:
                self.confirmed = 0
            if self.confirmed >= self.confirm_frames:
                self.stage = "turn"
                self.confirmed = 0
                self.last_q0 = float(feedback[0])
                self.last_q10 = float(feedback[10])
                # Rotation starts now, but q15 remains fully closed.
                output[list(THUMB_ORIENTATION_IDX)] = orientation_target
            return np.clip(output, 0, 255).astype(np.int32).tolist(), True, False

        if self.stage == "turn":
            self.last_q0 = float(feedback[0])
            self.last_q10 = float(feedback[10])
            turn_complete = bool(
                self.last_q0 <= self.turn_release_q0
                and self.last_q10 <= self.turn_release_q10
            )
            if turn_complete:
                self.confirmed += 1
            else:
                self.confirmed = 0
            if self.confirmed < self.turn_confirm_frames:
                return np.clip(output, 0, 255).astype(np.int32).tolist(), True, False

            self.released = True
            self.stage = "released"
            # Keep q15 closed on the exact release frame.
            output[list(THUMB_ORIENTATION_IDX)] = orientation_target
            return np.clip(output, 0, 255).astype(np.int32).tolist(), False, True

        raise RuntimeError(f"unexpected thumb gate stage {self.stage!r}")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT)
    ap.add_argument("--camera-index", type=int, default=2)
    ap.add_argument("--camera-width", type=int, default=640)
    ap.add_argument("--camera-height", type=int, default=480)
    ap.add_argument("--camera-fps", type=int, default=30)
    ap.add_argument("--camera-fourcc", default="MJPG")
    ap.add_argument("--side", choices=("right", "left"), default="right")
    ap.add_argument("--device", choices=("cuda", "cpu"), default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--rate", type=float, default=5.0)
    ap.add_argument(
        "--n-action-steps",
        type=int,
        default=10,
        help="number of predicted ACT chunk actions to execute before re-planning",
    )
    ap.add_argument(
        "--temporal-ensemble-coeff",
        type=float,
        default=None,
        help=(
            "enable ACT temporal ensembling across overlapping chunks; 0.01 is "
            "the original ACT default and requires --n-action-steps 1"
        ),
    )
    ap.add_argument(
        "--chunk-boundary-blend-frames",
        type=int,
        default=0,
        help=(
            "cross-fade this many commands at each ACT chunk boundary; "
            "0 disables boundary-only smoothing"
        ),
    )
    ap.add_argument(
        "--chunk-boundary-blend-thumb-only",
        action="store_true",
        help=(
            "apply chunk-boundary cross-fade only to thumb q0/q5/q10/q15; "
            "the four fingers use each new chunk without blending"
        ),
    )
    ap.add_argument(
        "--policy-handoff-checkpoint-dir",
        type=Path,
        default=None,
        help=(
            "preload a second ACT checkpoint; press P to move to its start pose, "
            "warm its history, and switch to it"
        ),
    )
    ap.add_argument(
        "--policy-handoff-start-pose",
        type=parse_g20_pose,
        default=None,
        help="absolute 20-D SDK pose used as the feedback-gated P-key handoff target",
    )
    ap.add_argument("--policy-handoff-n-action-steps", type=int, default=5)
    ap.add_argument("--policy-handoff-range-step", type=int, default=10)
    ap.add_argument("--policy-handoff-tolerance", type=float, default=12.0)
    ap.add_argument("--policy-handoff-confirm-frames", type=int, default=3)
    ap.add_argument("--policy-handoff-warmup-seconds", type=float, default=3.0)
    ap.add_argument("--policy-handoff-timeout", type=float, default=12.0)
    ap.add_argument(
        "--policy-handoff-current-limit",
        type=int,
        default=None,
        help="optional torque/current setting published when the second policy starts",
    )
    ap.add_argument(
        "--policy-handoff-speed-limit",
        type=int,
        default=None,
        help="optional speed setting published when the second policy starts",
    )
    ap.add_argument(
        "--cube-ready-profile",
        type=Path,
        default=None,
        help=(
            "enable a pre-arm ArUco pose gate using this calibration JSON; "
            "press G while DISARMED with the cube correctly on the fingers to "
            "create or replace it"
        ),
    )
    ap.add_argument("--cube-ready-center-tolerance", type=float, default=0.06)
    ap.add_argument("--cube-ready-scale-tolerance", type=float, default=0.30)
    ap.add_argument(
        "--cube-ready-ignore-scale",
        action="store_true",
        help=(
            "match the cube-ready pose by marker center only; useful when the "
            "cube face or viewing angle changes its apparent size"
        ),
    )
    ap.add_argument("--cube-ready-confirm-frames", type=int, default=5)
    ap.add_argument("--cube-ready-min-markers", type=int, default=3)
    ap.add_argument(
        "--cube-contact-gate",
        action="store_true",
        help=(
            "allow arming only after fresh touch data reports stable cube "
            "contact on the requested number of fingers; no G calibration"
        ),
    )
    ap.add_argument("--cube-contact-threshold", type=float, default=5.0)
    ap.add_argument("--cube-contact-min-fingers", type=int, default=1)
    ap.add_argument("--cube-contact-hold-seconds", type=float, default=0.15)
    ap.add_argument("--max-range-step", type=int, default=2)
    ap.add_argument("--ema-alpha", type=float, default=0.7)
    ap.add_argument(
        "--thumb-tip-before-turn",
        action="store_true",
        help=(
            "during the recorded A4 pre-turn pose, hold q0/q5/q10 at "
            "254/0/51 and command q15=0 until measured q15 is closed"
        ),
    )
    ap.add_argument("--thumb-tip-release-threshold", type=float, default=8.0)
    ap.add_argument("--thumb-tip-confirm-frames", type=int, default=3)
    ap.add_argument("--thumb-turn-release-q0", type=float, default=220.0)
    ap.add_argument("--thumb-turn-release-q10", type=float, default=10.0)
    ap.add_argument("--thumb-turn-confirm-frames", type=int, default=3)
    ap.add_argument(
        "--keyboard-thumb-bias",
        action="store_true",
        help=(
            "enable persistent live keyboard bias on one thumb channel: "
            "[ subtracts, ] adds, and backslash resets the bias"
        ),
    )
    ap.add_argument(
        "--keyboard-thumb-bias-joint",
        type=int,
        choices=THUMB_CONTROL_IDX,
        default=15,
        help="G20 thumb channel controlled by the keyboard bias; q15 is the tip",
    )
    ap.add_argument("--thumb-bias-step", type=int, default=5)
    ap.add_argument("--thumb-bias-limit", type=int, default=50)
    ap.add_argument("--thumb-bias-initial", type=int, default=0)
    ap.add_argument(
        "--keyboard-thumb-side-bias",
        action="store_true",
        help=(
            "enable a second persistent keyboard bias on q5 thumb side-swing: "
            "A moves left (subtract), D moves right (add), and S resets"
        ),
    )
    ap.add_argument("--thumb-side-bias-step", type=int, default=5)
    ap.add_argument("--thumb-side-bias-limit", type=int, default=80)
    ap.add_argument("--thumb-side-bias-initial", type=int, default=0)
    ap.add_argument(
        "--keyboard-action-library",
        type=Path,
        default=None,
        help=(
            "enable numbered action-library intervention while ARMED: keys 1-8 "
            "play an available primitive from its nearest measured-state frame, "
            "hold its endpoint, and key 9 returns control to ACT"
        ),
    )
    ap.add_argument(
        "--action-intervention-blend-frames",
        type=int,
        default=8,
        help="minimum measured-state blend frames before numbered playback",
    )
    ap.add_argument(
        "--thumb-final-push-offset",
        type=int,
        default=0,
        help=(
            "add this many q15 ticks only after the physical thumb is aligned "
            "in the final A4 push pose; 10 changes the recorded target 50 to 60"
        ),
    )
    ap.add_argument("--current-limit", type=int, default=20)
    ap.add_argument("--speed-limit", type=int, default=20)
    ap.add_argument(
        "--max-target-delta",
        type=float,
        default=80.0,
        help="disarm if any active target differs from observed state by more than this",
    )
    ap.add_argument(
        "--max-raw-overshoot",
        type=float,
        default=20.0,
        help="disarm if raw ACT output leaves [-margin, 255+margin]",
    )
    ap.add_argument(
        "--max-active-seconds",
        type=float,
        default=10.0,
        help="automatically disarm after this many seconds; <=0 disables",
    )
    ap.add_argument(
        "--auto-stop-endpoint-profile",
        type=Path,
        default=None,
        help=(
            "auto-disarm after departing the start and then stably reaching any "
            "successful demonstration endpoint stored in this JSON profile"
        ),
    )
    ap.add_argument("--auto-stop-endpoint-tolerance", type=float, default=12.0)
    ap.add_argument("--auto-stop-endpoint-confirm-frames", type=int, default=10)
    ap.add_argument("--auto-stop-min-active-seconds", type=float, default=8.0)
    ap.add_argument("--auto-stop-departure-delta", type=float, default=20.0)
    ap.add_argument(
        "--hold-on-disarm",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="publish fresh measured joint positions once when manually or automatically disarming",
    )
    ap.add_argument("--state-timeout", type=float, default=5.0)
    ap.add_argument("--state-stale-seconds", type=float, default=0.5)
    ap.add_argument("--log-period", type=float, default=0.5)
    ap.add_argument(
        "--minimal-overlay",
        action="store_true",
        help=(
            "show only the mode, key help, and safety-critical messages in the "
            "camera window; hide routine ACT, touch, endpoint, and target-delta "
            "diagnostics without changing terminal logs"
        ),
    )
    ap.add_argument(
        "--record-rated-attempts",
        action="store_true",
        help="record each armed attempt, then require a 0/5/1 quality key",
    )
    ap.add_argument(
        "--rated-output-dir",
        type=Path,
        default=REPO_ROOT / "data" / "self_imitation",
    )
    ap.add_argument("--record-rate", type=float, default=30.0)
    ap.add_argument("--jpeg-quality", type=int, default=92)
    ap.add_argument(
        "--contact-on-threshold",
        type=float,
        default=20.0,
        help="grams required to turn a finger contact on",
    )
    ap.add_argument(
        "--contact-off-threshold",
        type=float,
        default=10.0,
        help="grams below which an existing finger contact turns off",
    )
    ap.add_argument("--min-contact-fingers", type=int, default=3)
    ap.add_argument(
        "--require-thumb-contact",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="require the thumb among the minimum contacting fingers",
    )
    ap.add_argument("--contact-hold-seconds", type=float, default=0.5)
    ap.add_argument("--touch-stale-seconds", type=float, default=0.5)
    ap.add_argument(
        "--ignore-touch",
        action="store_true",
        help=(
            "do not display touch or use it for stopping, success, or rating; "
            "incompatible with a mass-contact tactile checkpoint"
        ),
    )
    ap.add_argument(
        "--require-touch-for-score-one",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="downgrade human score 1 to 0.5 unless the touch gate is met",
    )
    ap.add_argument(
        "--stop-on-contact-success",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="stop an armed rated attempt after the stable touch gate is met",
    )
    ap.add_argument(
        "--reset-after-rating",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="after a 0/5/1 rating, ramp back to the standard G20 open pose",
    )
    ap.add_argument("--reset-range-step", type=int, default=15)
    ap.add_argument("--reset-tolerance", type=int, default=3)
    ap.add_argument("--enable-motion", action="store_true")
    args = ap.parse_args()
    args.checkpoint_dir = args.checkpoint_dir.expanduser().resolve()
    if args.policy_handoff_checkpoint_dir is not None:
        args.policy_handoff_checkpoint_dir = (
            args.policy_handoff_checkpoint_dir.expanduser().resolve()
        )
    args.rated_output_dir = args.rated_output_dir.expanduser().resolve()
    if args.cube_ready_profile is not None:
        args.cube_ready_profile = args.cube_ready_profile.expanduser().resolve()
    if args.auto_stop_endpoint_profile is not None:
        args.auto_stop_endpoint_profile = (
            args.auto_stop_endpoint_profile.expanduser().resolve()
        )
    if args.keyboard_action_library is not None:
        args.keyboard_action_library = (
            args.keyboard_action_library.expanduser().resolve()
        )
    if (
        args.rate <= 0
        or args.record_rate <= 0
        or args.max_range_step < 0
        or args.reset_range_step <= 0
        or args.reset_tolerance < 0
        or args.n_action_steps <= 0
        or args.contact_off_threshold < 0
        or args.contact_on_threshold <= 0
        or not 1 <= args.min_contact_fingers <= 5
        or args.contact_hold_seconds < 0
        or args.touch_stale_seconds <= 0
        or args.cube_ready_center_tolerance < 0
        or args.cube_ready_scale_tolerance < 0
        or args.cube_ready_confirm_frames <= 0
        or not 2 <= args.cube_ready_min_markers <= 4
        or args.cube_contact_threshold <= 0
        or not 1 <= args.cube_contact_min_fingers <= 5
        or args.cube_contact_hold_seconds < 0
        or args.thumb_tip_release_threshold < 0
        or args.thumb_tip_confirm_frames <= 0
        or not 0 <= args.thumb_turn_release_q0 <= 255
        or not 0 <= args.thumb_turn_release_q10 <= 255
        or args.thumb_turn_confirm_frames <= 0
        or args.thumb_bias_step <= 0
        or not 0 <= args.thumb_bias_limit <= 100
        or abs(args.thumb_bias_initial) > args.thumb_bias_limit
        or args.thumb_side_bias_step <= 0
        or not 0 <= args.thumb_side_bias_limit <= 100
        or abs(args.thumb_side_bias_initial) > args.thumb_side_bias_limit
        or args.action_intervention_blend_frames < 0
        or abs(args.thumb_final_push_offset) > 50
        or args.auto_stop_endpoint_tolerance < 0
        or args.auto_stop_endpoint_confirm_frames <= 0
        or args.auto_stop_min_active_seconds < 0
        or args.auto_stop_departure_delta <= 0
        or args.policy_handoff_n_action_steps <= 0
        or args.policy_handoff_range_step <= 0
        or args.policy_handoff_tolerance < 0
        or args.policy_handoff_confirm_frames <= 0
        or args.policy_handoff_warmup_seconds < 0
        or args.policy_handoff_timeout <= 0
    ):
        ap.error("--rate must be positive and --max-range-step must be non-negative")
    if args.contact_off_threshold >= args.contact_on_threshold:
        ap.error("--contact-off-threshold must be lower than --contact-on-threshold")
    if (
        args.keyboard_thumb_bias
        and args.keyboard_thumb_side_bias
        and args.keyboard_thumb_bias_joint == 5
    ):
        ap.error(
            "q5 cannot use both --keyboard-thumb-bias and "
            "--keyboard-thumb-side-bias"
        )
    if not 0.0 <= args.ema_alpha < 1.0:
        ap.error("--ema-alpha must be in [0, 1)")
    if not 0 <= args.chunk_boundary_blend_frames <= args.n_action_steps:
        ap.error(
            "--chunk-boundary-blend-frames must be in "
            "[0, --n-action-steps]"
        )
    if args.temporal_ensemble_coeff is not None:
        if not np.isfinite(args.temporal_ensemble_coeff):
            ap.error("--temporal-ensemble-coeff must be finite")
        if args.n_action_steps != 1:
            ap.error(
                "--temporal-ensemble-coeff requires --n-action-steps 1 so ACT "
                "can re-plan every frame"
            )
        if args.chunk_boundary_blend_frames:
            ap.error(
                "--chunk-boundary-blend-frames cannot be combined with "
                "--temporal-ensemble-coeff"
            )
    if args.cube_ready_profile is not None and args.cube_contact_gate:
        ap.error(
            "choose either --cube-ready-profile or --cube-contact-gate, not both"
        )
    if args.cube_contact_gate and args.ignore_touch:
        ap.error("--cube-contact-gate cannot be combined with --ignore-touch")
    if len(args.camera_fourcc) != 4:
        ap.error("--camera-fourcc must contain four characters")
    if args.record_rated_attempts and not args.enable_motion:
        ap.error("--record-rated-attempts requires --enable-motion")
    if args.keyboard_action_library is not None and args.max_range_step <= 0:
        ap.error("--keyboard-action-library requires --max-range-step > 0")
    handoff_checkpoint_set = args.policy_handoff_checkpoint_dir is not None
    handoff_pose_set = args.policy_handoff_start_pose is not None
    if handoff_checkpoint_set != handoff_pose_set:
        ap.error(
            "--policy-handoff-checkpoint-dir and --policy-handoff-start-pose "
            "must be supplied together"
        )
    for name in ("policy_handoff_current_limit", "policy_handoff_speed_limit"):
        value = getattr(args, name)
        if value is not None and not 1 <= value <= 255:
            ap.error(f"--{name.replace('_', '-')} must be in [1, 255]")
    if args.record_rated_attempts and handoff_checkpoint_set:
        ap.error(
            "--record-rated-attempts is not supported during a two-policy handoff"
        )
    if args.ignore_touch:
        args.require_touch_for_score_one = False
        args.stop_on_contact_success = False
    args.jpeg_quality = int(np.clip(args.jpeg_quality, 1, 100))
    return args


def load_policy(
    checkpoint: Path,
    device: str,
    n_action_steps: int = 10,
    temporal_ensemble_coeff: float | None = None,
) -> tuple[Any, Any, Any]:
    if not (checkpoint / "model.safetensors").is_file():
        raise RuntimeError(f"invalid checkpoint directory: {checkpoint}")
    overrides = ["--device", device, "--n_action_steps", str(n_action_steps)]
    if temporal_ensemble_coeff is not None:
        overrides.extend([
            "--temporal_ensemble_coeff",
            str(temporal_ensemble_coeff),
        ])
    cfg = PreTrainedConfig.from_pretrained(
        str(checkpoint),
        local_files_only=True,
        cli_overrides=overrides,
    )
    expected_inputs = {"observation.state", "observation.images.scene"}
    if set(cfg.input_features) != expected_inputs:
        raise RuntimeError(
            f"checkpoint inputs are {list(cfg.input_features)}; expected {sorted(expected_inputs)}"
        )
    policy_cls = get_policy_class(cfg.type)
    policy = policy_cls.from_pretrained(
        str(checkpoint), config=cfg, local_files_only=True
    )
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        pretrained_path=str(checkpoint),
        preprocessor_overrides={"device_processor": {"device": device}},
        postprocessor_overrides={"device_processor": {"device": "cpu"}},
    )
    policy.eval()
    policy.reset()
    return policy, preprocessor, postprocessor


class G20VisualACTNode:
    def __init__(self, args: argparse.Namespace) -> None:
        import rclpy
        from rclpy.node import Node
        from sensor_msgs.msg import JointState
        from std_msgs.msg import String

        self.rclpy = rclpy
        self.JointState = JointState
        self.String = String
        self.node = Node("g20_visual_act_policy")
        self.state_topic = f"/cb_{args.side}_hand_state"
        self.mass_topic = f"/cb_{args.side}_hand_matrix_touch_mass"
        self.command_topic = f"/cb_{args.side}_hand_control_cmd"
        self.setting_topic = "/cb_hand_setting_cmd"
        self.command_pub = self.node.create_publisher(JointState, self.command_topic, 10)
        self.setting_pub = self.node.create_publisher(String, self.setting_topic, 10)
        self.node.create_subscription(JointState, self.state_topic, self._state_cb, 10)
        if not args.ignore_touch:
            self.node.create_subscription(String, self.mass_topic, self._mass_cb, 10)
        self.last_state: Optional[list[int]] = None
        self.last_state_time = 0.0
        self.last_mass_values: Optional[list[float]] = None
        self.contact_tracker = TouchContactTracker(
            args.contact_on_threshold,
            args.contact_off_threshold,
            args.min_contact_fingers,
            args.require_thumb_contact,
            args.contact_hold_seconds,
        )

    def _state_cb(self, msg: Any) -> None:
        if len(msg.position) < JOINT_COUNT:
            return
        self.last_state = np.clip(
            np.rint(np.asarray(msg.position[:JOINT_COUNT])), 0, 255
        ).astype(np.int32).tolist()
        self.last_state_time = time.monotonic()

    def _mass_cb(self, msg: Any) -> None:
        try:
            data = json.loads(msg.data)
        except (json.JSONDecodeError, TypeError):
            return
        if not isinstance(data, dict):
            return
        values = [flatten_sum(data.get(key, 0.0)) for key in MASS_KEYS]
        self.last_mass_values = values
        self.contact_tracker.update(values, time.monotonic())

    def begin_contact_attempt(self, now: float) -> None:
        self.contact_tracker.begin_attempt(now)

    def contact_snapshot(self, now: float, stale_seconds: float) -> dict[str, Any]:
        return self.contact_tracker.snapshot(now, stale_seconds)

    def end_contact_attempt(self) -> None:
        self.contact_tracker.end_attempt()

    def spin(self, timeout: float = 0.0) -> None:
        # ACT inference and camera reads can temporarily block longer than one
        # ROS publish period.  Drain a small callback burst so a queued touch
        # callback cannot leave the joint-state timestamp artificially stale.
        self.rclpy.spin_once(self.node, timeout_sec=timeout)
        for _ in range(31):
            self.rclpy.spin_once(self.node, timeout_sec=0.0)

    def wait_ready(self, timeout: float, require_subscriber: bool) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self.spin(0.05)
            state_ready = self.last_state is not None
            subscriber_ready = (
                self.command_pub.get_subscription_count() > 0
                and self.setting_pub.get_subscription_count() > 0
            )
            if state_ready and (subscriber_ready or not require_subscriber):
                return True
        return False

    def wait_touch(self, timeout: float) -> bool:
        """Wait for one tactile mass sample without publishing anything."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self.spin(0.05)
            if self.last_mass_values is not None:
                return True
        return False

    def state_is_fresh(self, stale_seconds: float) -> bool:
        return self.last_state is not None and (
            time.monotonic() - self.last_state_time <= stale_seconds
        )

    def publish_limits(
        self,
        side: str,
        current_limit: int,
        speed_limit: int,
    ) -> None:
        settings = (
            ("set_max_torque_limits", "torque", [int(current_limit)] * 5),
            ("set_speed", "speed", [int(speed_limit)] * 5),
        )
        for setting_cmd, key, values in settings:
            msg = self.String()
            msg.data = json.dumps(
                {"setting_cmd": setting_cmd, "params": {"hand_type": side, key: values}}
            )
            self.setting_pub.publish(msg)

    def publish_settings(self, args: argparse.Namespace) -> None:
        self.publish_limits(args.side, args.current_limit, args.speed_limit)

    def publish_pose(self, pose: list[int]) -> None:
        msg = self.JointState()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.position = [float(value) for value in pose]
        msg.velocity = [0.0] * JOINT_COUNT
        msg.effort = [0.0] * JOINT_COUNT
        self.command_pub.publish(msg)

    def close(self) -> None:
        self.node.destroy_node()


class RatedAttemptRecorder:
    """Write policy attempts in the same camera/state/action format as T3."""

    def __init__(self, args: argparse.Namespace) -> None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = args.rated_output_dir / f"{stamp}_act_self_imitation"
        self.session_dir.mkdir(parents=True, exist_ok=False)
        self.args = args
        self.episode_index = 0
        self.episode_dir: Optional[Path] = None
        self.image_dir: Optional[Path] = None
        self.samples_file: Optional[Any] = None
        self.sample_index = 0
        self.started_mono = 0.0
        self.last_sample_mono = 0.0
        self.awaiting_score = False
        self.stop_reason = ""
        self.contact_summary: Optional[dict[str, Any]] = None
        session = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "task_id": "orientation_grasp_self_imitation",
            "hand_type": args.side,
            "hand_model": "g20_palm_touch",
            "source": "visual_act_human_rated",
            "checkpoint": str(args.checkpoint_dir),
            "rating_keys": {"0": 0.0, "5": 0.5, "1": 1.0},
            "camera": {
                "source": args.camera_index,
                "width": args.camera_width,
                "height": args.camera_height,
                "fps": args.camera_fps,
                "jpeg_quality": args.jpeg_quality,
            },
            "schema": {
                "joint_pos": "20 floats, observed SDK units 0-255",
                "last_action": "20 floats, safety-limited command actually published",
                "control_source": "act_policy or action_library_N",
                "thumb_tip_bias": "persistent additive q15 keyboard bias in SDK ticks",
                "thumb_side_bias": "persistent additive q5 keyboard bias in SDK ticks",
                "mass_values": "6 floats in grams: thumb,index,middle,ring,little,palm",
                "contact_6": "hysteretic contacts for thumb,index,middle,ring,little,palm",
                "quality_score": "episode-level human rating: 0, 0.5, or 1",
            },
            "args": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
        }
        (self.session_dir / "session.json").write_text(
            json.dumps(session, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"[rated_recorder] session: {self.session_dir}", flush=True)

    @property
    def recording(self) -> bool:
        return self.samples_file is not None

    def start(self) -> bool:
        if self.recording or self.awaiting_score:
            return False
        self.episode_dir = self.session_dir / f".pending_episode_{self.episode_index:03d}"
        self.image_dir = self.episode_dir / "images"
        self.image_dir.mkdir(parents=True, exist_ok=False)
        self.samples_file = (self.episode_dir / "samples.jsonl").open(
            "w", encoding="utf-8", buffering=1
        )
        self.sample_index = 0
        self.started_mono = time.monotonic()
        self.last_sample_mono = 0.0
        self.stop_reason = ""
        self.contact_summary = None
        print(f"[rated_recorder] REC episode_{self.episode_index:03d}", flush=True)
        return True

    def add(
        self,
        frame: np.ndarray,
        state: list[int],
        command: list[int],
        now_mono: float,
        mass_values: Optional[list[float]],
        contact_6: Optional[list[int]],
        mass_age: Optional[float],
        control_source: str = "act_policy",
        thumb_tip_bias: int = 0,
        thumb_side_bias: int = 0,
    ) -> None:
        if not self.recording or self.episode_dir is None or self.image_dir is None:
            return
        if now_mono - self.last_sample_mono + 1e-6 < 1.0 / self.args.record_rate:
            return
        image_name = f"{self.sample_index:06d}.jpg"
        image_path = self.image_dir / image_name
        if not cv2.imwrite(
            str(image_path),
            frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), self.args.jpeg_quality],
        ):
            print(f"[rated_recorder] dropped image {image_path}", file=sys.stderr)
            return
        sample = {
            "index": self.sample_index,
            "episode": self.episode_index,
            "timestamp": time.time(),
            "elapsed": now_mono - self.started_mono,
            "task_id": "orientation_grasp_self_imitation",
            "image_path": f"images/{image_name}",
            "joint_pos": [float(value) for value in state],
            "joint_vel": [0.0] * JOINT_COUNT,
            "joint_effort": [0.0] * JOINT_COUNT,
            "mass_values": mass_values,
            "contact_6": contact_6,
            "last_action": [float(value) for value in command],
            "has_matrix": False,
            "ages": {"mass": mass_age},
            "policy_checkpoint": str(self.args.checkpoint_dir),
            "control_source": str(control_source),
            "thumb_tip_bias": int(thumb_tip_bias),
            "thumb_side_bias": int(thumb_side_bias),
        }
        assert self.samples_file is not None
        self.samples_file.write(
            json.dumps(sample, ensure_ascii=False, separators=(",", ":")) + "\n"
        )
        self.sample_index += 1
        self.last_sample_mono = now_mono

    def stop(
        self, reason: str, contact_summary: Optional[dict[str, Any]] = None
    ) -> bool:
        if not self.recording:
            return False
        assert self.samples_file is not None and self.episode_dir is not None
        self.samples_file.close()
        self.samples_file = None
        self.stop_reason = reason
        self.contact_summary = contact_summary
        if self.sample_index == 0:
            shutil.rmtree(self.episode_dir)
            self.episode_dir = None
            self.image_dir = None
            print("[rated_recorder] empty attempt discarded", flush=True)
            return False
        self.awaiting_score = True
        self._write_episode_metadata(None)
        print(
            f"[rated_recorder] STOP {self.sample_index} samples; press 0, 5, or 1 to rate",
            flush=True,
        )
        return True

    def _write_episode_metadata(self, score: Optional[float]) -> None:
        assert self.episode_dir is not None
        metadata = {
            "episode_index": self.episode_index,
            "quality_score": score,
            "rating_status": "rated" if score is not None else "pending",
            "sample_count": self.sample_index,
            "stop_reason": self.stop_reason,
            "checkpoint": str(self.args.checkpoint_dir),
            "contact_summary": self.contact_summary,
        }
        (self.episode_dir / "episode.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    def rate(
        self, score: float, human_score: Optional[float] = None
    ) -> Optional[Path]:
        if not self.awaiting_score or self.episode_dir is None:
            return None
        if score not in (0.0, 0.5, 1.0):
            raise ValueError(f"invalid score {score}")
        self._write_episode_metadata(score)
        metadata_path = self.episode_dir / "episode.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["human_quality_score"] = score if human_score is None else human_score
        metadata["touch_adjusted_score"] = score
        metadata_path.write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        final_dir = self.session_dir / f"episode_{self.episode_index:03d}"
        self.episode_dir.rename(final_dir)
        print(
            f"[rated_recorder] SAVED episode_{self.episode_index:03d} score={score:g} "
            f"samples={self.sample_index}",
            flush=True,
        )
        self.episode_index += 1
        self.episode_dir = None
        self.image_dir = None
        self.awaiting_score = False
        self.sample_index = 0
        self.contact_summary = None
        return final_dir

    def close(self) -> None:
        if self.recording:
            self.stop("process_exit")
        if self.awaiting_score:
            print(
                f"[rated_recorder] leaving unrated attempt in {self.episode_dir}",
                flush=True,
            )


def open_camera(args: argparse.Namespace) -> Any:
    camera = cv2.VideoCapture(args.camera_index, cv2.CAP_V4L2)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, args.camera_width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, args.camera_height)
    camera.set(cv2.CAP_PROP_FPS, args.camera_fps)
    camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*args.camera_fourcc))
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not camera.isOpened():
        camera.release()
        raise RuntimeError(f"could not open camera index {args.camera_index}")
    for _ in range(20):
        ok, frame = camera.read()
        if ok and frame is not None:
            return camera
    camera.release()
    raise RuntimeError(f"camera {args.camera_index} opened but returned no frames")


def _manifest_offsets(
    manifest_path: Path,
    key: str,
    *, allowed_lengths: tuple[int, ...] | None = None,
) -> tuple[int, ...] | None:
    if not manifest_path.is_file():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        offsets = tuple(int(value) for value in manifest.get(key, [0]))
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return None
    if (
        offsets
        and offsets[-1] == 0
        and all(value >= 0 for value in offsets)
        and (allowed_lengths is None or len(offsets) in allowed_lengths)
    ):
        return offsets
    return None


def checkpoint_history_offsets(checkpoint: Path) -> tuple[int, ...]:
    """Read temporal image offsets near a checkpoint or its reused dataset."""
    candidates = [
        parent / "dataset" / "g20_source_manifest.json"
        for parent in checkpoint.parents
    ]

    # A training-only artifact may reference a dataset under another artifact
    # root.  LeRobot saves that root in every pretrained_model/train_config.json.
    train_config_path = checkpoint / "train_config.json"
    if train_config_path.is_file():
        try:
            train_config = json.loads(train_config_path.read_text(encoding="utf-8"))
            reused_root = train_config.get("dataset", {}).get("root")
            if reused_root:
                reused_root = Path(reused_root).expanduser()
                if not reused_root.is_absolute():
                    reused_root = train_config_path.parent / reused_root
                candidates.append(reused_root / "g20_source_manifest.json")
        except (OSError, json.JSONDecodeError, TypeError):
            pass

    for manifest_path in candidates:
        offsets = _manifest_offsets(
            manifest_path,
            "history_frame_offsets",
            allowed_lengths=(1, 4, 6),
        )
        if offsets is not None:
            return offsets
    return (0,)


def checkpoint_state_history_offsets(checkpoint: Path) -> tuple[int, ...]:
    """Read exact joint-state history offsets stored beside a checkpoint."""
    candidates = [
        parent / "dataset" / "g20_source_manifest.json"
        for parent in checkpoint.parents
    ]
    train_config_path = checkpoint / "train_config.json"
    if train_config_path.is_file():
        try:
            train_config = json.loads(train_config_path.read_text(encoding="utf-8"))
            reused_root = train_config.get("dataset", {}).get("root")
            if reused_root:
                reused_root = Path(reused_root).expanduser()
                if not reused_root.is_absolute():
                    reused_root = train_config_path.parent / reused_root
                candidates.append(reused_root / "g20_source_manifest.json")
        except (OSError, json.JSONDecodeError, TypeError):
            pass
    for manifest_path in candidates:
        offsets = _manifest_offsets(manifest_path, "state_history_offsets")
        if offsets is not None:
            return offsets
    return (0,)


def checkpoint_tactile_mode(checkpoint: Path) -> str:
    """Read the opt-in coarse tactile layout stored beside a checkpoint."""
    candidates = [
        parent / "dataset" / "g20_source_manifest.json"
        for parent in checkpoint.parents
    ]
    train_config_path = checkpoint / "train_config.json"
    if train_config_path.is_file():
        try:
            train_config = json.loads(train_config_path.read_text(encoding="utf-8"))
            reused_root = train_config.get("dataset", {}).get("root")
            if reused_root:
                reused_root = Path(reused_root).expanduser()
                if not reused_root.is_absolute():
                    reused_root = train_config_path.parent / reused_root
                candidates.append(reused_root / "g20_source_manifest.json")
        except (OSError, json.JSONDecodeError, TypeError):
            pass
    for manifest_path in candidates:
        if not manifest_path.is_file():
            continue
        try:
            mode = str(
                json.loads(manifest_path.read_text(encoding="utf-8")).get(
                    "tactile_mode", "none"
                )
            )
        except (OSError, json.JSONDecodeError, TypeError):
            continue
        if mode in ("none", "mass-contact"):
            return mode
    return "none"


def history_mosaic_bgr(
    frames: deque[np.ndarray], offsets: tuple[int, ...]
) -> np.ndarray:
    """Match the converter's single-frame or chronological history image."""
    if not frames:
        raise RuntimeError("camera history is empty")
    selected = [frames[max(0, len(frames) - 1 - offset)] for offset in offsets]
    if len(selected) == 1:
        return cv2.resize(selected[0], (320, 240), interpolation=cv2.INTER_AREA)
    rows, columns = (2, 2) if len(selected) == 4 else (2, 3)
    mosaic_rows: list[np.ndarray] = []
    for row_index in range(rows):
        tiles: list[np.ndarray] = []
        y0 = round(row_index * 240 / rows)
        y1 = round((row_index + 1) * 240 / rows)
        for column_index in range(columns):
            x0 = round(column_index * 320 / columns)
            x1 = round((column_index + 1) * 320 / columns)
            frame = selected[row_index * columns + column_index]
            tiles.append(
                cv2.resize(
                    frame,
                    (x1 - x0, y1 - y0),
                    interpolation=cv2.INTER_AREA,
                )
            )
        mosaic_rows.append(np.concatenate(tiles, axis=1))
    return np.concatenate(mosaic_rows, axis=0)


def make_observation(
    frames: deque[np.ndarray],
    state: list[int],
    history_offsets: tuple[int, ...] = (0,),
    states: Optional[deque[list[int]]] = None,
    state_history_offsets: tuple[int, ...] = (0,),
    tactile_mode: str = "none",
    mass_values: Optional[list[float]] = None,
    contact_threshold: float = 20.0,
) -> dict[str, torch.Tensor]:
    resized = history_mosaic_bgr(frames, history_offsets)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    image = torch.from_numpy(np.ascontiguousarray(rgb)).float().div_(255.0)
    image = image.permute(2, 0, 1).unsqueeze(0)
    state_frames = states if states else deque([list(state)])
    selected_states = [
        state_frames[max(0, len(state_frames) - 1 - offset)]
        for offset in state_history_offsets
    ]
    state_vector = np.concatenate(
        [np.asarray(value, dtype=np.float32)[:JOINT_COUNT] for value in selected_states]
    )
    if tactile_mode == "mass-contact":
        mass = np.zeros(6, dtype=np.float32)
        if mass_values is not None:
            observed = np.asarray(mass_values[:6], dtype=np.float32)
            if observed.shape == (6,) and np.all(np.isfinite(observed)):
                mass = np.maximum(observed, 0.0)
        contact = (mass >= float(contact_threshold)).astype(np.float32)
        state_vector = np.concatenate((state_vector, mass, contact))
    elif tactile_mode != "none":
        raise ValueError(f"unsupported tactile mode: {tactile_mode}")
    state_tensor = torch.as_tensor(state_vector, dtype=torch.float32).unsqueeze(0)
    return {
        "observation.state": state_tensor,
        "observation.images.scene": image,
    }


def draw_status(
    frame: np.ndarray,
    armed: bool,
    motion_enabled: bool,
    raw: Optional[np.ndarray],
    message: str,
    awaiting_score: bool = False,
    resetting: bool = False,
    contact_summary: Optional[dict[str, Any]] = None,
    cube_ready_text: str | None = None,
    minimal_overlay: bool = False,
    handoff_enabled: bool = False,
    handoff_status: str | None = None,
    keyboard_thumb_bias_joint: int | None = None,
    keyboard_thumb_bias: int = 0,
    keyboard_thumb_side_bias: int | None = None,
    action_intervention_enabled: bool = False,
    action_intervention_status: str | None = None,
) -> None:
    if handoff_status is not None:
        mode = f"POLICY HANDOFF / {handoff_status}"
        color = (0, 165, 255)
    elif resetting:
        mode = "RESETTING TO OPEN POSE"
        color = (255, 180, 0)
    elif armed and action_intervention_status is not None:
        mode = f"ARMED / LIBRARY {action_intervention_status}"
        color = (0, 165, 255)
    else:
        mode = "ARMED / PUBLISHING" if armed else "DISARMED / NO COMMANDS"
        color = (0, 0, 255) if armed else (0, 200, 255)
    cv2.putText(frame, mode, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
    help_text = "SPACE arm/disarm | R reset"
    if action_intervention_enabled:
        help_text += " | 1-8 library | 9 policy"
    if handoff_enabled:
        help_text += " | P switch policy"
    help_text += " | Q/ESC quit"
    cv2.putText(
        frame,
        help_text,
        (15, 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (255, 255, 255),
        2,
    )
    if (
        keyboard_thumb_bias_joint is not None
        or keyboard_thumb_side_bias is not None
    ):
        bias_parts: list[str] = []
        if keyboard_thumb_bias_joint is not None:
            bias_parts.append(
                f"TIP q{keyboard_thumb_bias_joint}={keyboard_thumb_bias:+d}"
                "  [ -  ] +  \\ zero"
            )
        if keyboard_thumb_side_bias is not None:
            bias_parts.append(
                f" | SIDE q5={keyboard_thumb_side_bias:+d}"
                "  A left  D right  S zero"
            )
        bias_text = "".join(bias_parts).removeprefix(" | ")
        cv2.putText(
            frame,
            bias_text,
            (15, 86),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45 if keyboard_thumb_side_bias is not None else 0.55,
            (255, 220, 0),
            2,
        )
    if not motion_enabled:
        cv2.putText(
            frame,
            "DRY RUN (--enable-motion absent)",
            (15, 86),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (0, 255, 255),
            2,
        )
    if raw is not None and not minimal_overlay:
        text = f"ACT raw min/max: {float(raw.min()):.1f}/{float(raw.max()):.1f}"
        cv2.putText(frame, text, (15, 114), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    if awaiting_score:
        cv2.putText(
            frame,
            "RATE ATTEMPT: 0 bad | 5 partial | 1 good",
            (15, 144),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 0),
            2,
        )
    if contact_summary is not None and not minimal_overlay:
        count = int(contact_summary["finger_contact_count"])
        minimum = int(contact_summary["minimum_fingers"])
        thumb_ok = bool(contact_summary["thumb_contact"])
        stable = bool(contact_summary["success_gate_met"])
        fresh = bool(contact_summary["touch_fresh"])
        held = float(contact_summary["continuous_contact_seconds"])
        if not fresh:
            touch_text = "TOUCH: no fresh data"
            touch_color = (0, 0, 255)
        else:
            touch_text = (
                f"TOUCH {count}/{minimum} thumb={'Y' if thumb_ok else 'N'} "
                f"hold={held:.1f}s {'PASS' if stable else 'WAIT'}"
            )
            touch_color = (0, 220, 0) if stable else (0, 200, 255)
        cv2.putText(
            frame,
            touch_text,
            (15, 174),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            touch_color,
            2,
        )
    if cube_ready_text is not None and not minimal_overlay:
        is_ready = (
            cube_ready_text.startswith("CUBE READY")
            or cube_ready_text.startswith("START CONTACT READY")
        )
        ready_color = (0, 220, 0) if is_ready else (0, 200, 255)
        cv2.putText(
            frame,
            cube_ready_text[:100],
            (15, frame.shape[0] - 44),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            ready_color,
            2,
        )
    routine_message = message.startswith("max target delta")
    if message and (not minimal_overlay or not routine_message):
        cv2.putText(frame, message[:80], (15, frame.shape[0] - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 2)


def main() -> int:
    args = parse_args()
    if args.enable_motion and os.environ.get("HW_ENABLE_TOKEN") != "1":
        print(
            "[visual_act] refusing motion: set HW_ENABLE_TOKEN=1 manually and keep --enable-motion",
            file=sys.stderr,
        )
        return 2

    import rclpy

    rclpy.init(args=None)
    ros: Optional[G20VisualACTNode] = None
    camera = None
    recorder: Optional[RatedAttemptRecorder] = None
    try:
        ros = G20VisualACTNode(args)
        if not ros.wait_ready(args.state_timeout, require_subscriber=args.enable_motion):
            raise RuntimeError(
                f"no fresh state on {ros.state_topic}"
                + (
                    f", or SDK subscribers are missing on {ros.command_topic} / "
                    f"{ros.setting_topic}"
                    if args.enable_motion
                    else ""
                )
            )
        print(f"[visual_act] loading {args.checkpoint_dir}", flush=True)
        policy, preprocessor, postprocessor = load_policy(
            args.checkpoint_dir,
            args.device,
            args.n_action_steps,
            args.temporal_ensemble_coeff,
        )
        chunk_blend_indices = (
            THUMB_CONTROL_IDX
            if args.chunk_boundary_blend_thumb_only
            else None
        )
        chunk_blender = ChunkBoundaryBlender(
            args.n_action_steps,
            args.chunk_boundary_blend_frames,
            chunk_blend_indices,
        )
        action_intervention: ActionLibraryIntervention | None = None
        if args.keyboard_action_library is not None:
            try:
                intervention_library = ActionLibrary.load(
                    args.keyboard_action_library
                )
            except (OSError, ValueError, KeyError) as exc:
                raise RuntimeError(
                    "cannot load keyboard intervention action library "
                    f"{args.keyboard_action_library}: {exc}"
                ) from exc
            action_intervention = ActionLibraryIntervention(
                intervention_library,
                max_step=args.max_range_step,
                blend_frames=args.action_intervention_blend_frames,
            )
            available = sorted(
                action_id
                for action_id in intervention_library.primitives
                if 1 <= action_id <= 8
            )
            print(
                "[visual_act] numbered action intervention ready: "
                f"keys={available}; 9 returns to ACT; endpoint holds until 9",
                flush=True,
            )
        history_offsets = checkpoint_history_offsets(args.checkpoint_dir)
        state_history_offsets = checkpoint_state_history_offsets(args.checkpoint_dir)
        tactile_mode = checkpoint_tactile_mode(args.checkpoint_dir)
        if tactile_mode != "none":
            if args.ignore_touch:
                raise RuntimeError(
                    "tactile checkpoint cannot run with --ignore-touch"
                )
            if not ros.wait_touch(args.state_timeout):
                raise RuntimeError(
                    f"tactile checkpoint received no data on {ros.mass_topic}"
                )

        handoff_policy = None
        handoff_preprocessor = None
        handoff_postprocessor = None
        handoff_history_offsets: tuple[int, ...] = (0,)
        handoff_state_history_offsets: tuple[int, ...] = (0,)
        handoff_tactile_mode = "none"
        handoff_controller: PolicyHandoffController | None = None
        if args.policy_handoff_checkpoint_dir is not None:
            print(
                f"[visual_act] preloading P-key handoff policy "
                f"{args.policy_handoff_checkpoint_dir}",
                flush=True,
            )
            (
                handoff_policy,
                handoff_preprocessor,
                handoff_postprocessor,
            ) = load_policy(
                args.policy_handoff_checkpoint_dir,
                args.device,
                args.policy_handoff_n_action_steps,
                None,
            )
            handoff_history_offsets = checkpoint_history_offsets(
                args.policy_handoff_checkpoint_dir
            )
            handoff_state_history_offsets = checkpoint_state_history_offsets(
                args.policy_handoff_checkpoint_dir
            )
            handoff_tactile_mode = checkpoint_tactile_mode(
                args.policy_handoff_checkpoint_dir
            )
            assert args.policy_handoff_start_pose is not None
            handoff_controller = PolicyHandoffController(
                target_pose=args.policy_handoff_start_pose,
                tolerance=args.policy_handoff_tolerance,
                confirm_frames=args.policy_handoff_confirm_frames,
                warmup_seconds=args.policy_handoff_warmup_seconds,
                timeout_seconds=args.policy_handoff_timeout,
            )
            print(
                "[visual_act] P-key handoff ready: "
                f"visual_history={list(handoff_history_offsets)} "
                f"state_history={list(handoff_state_history_offsets)} "
                f"start_pose={list(map(int, args.policy_handoff_start_pose))}",
                flush=True,
            )

        max_visual_history = max(
            max(history_offsets),
            max(handoff_history_offsets),
        )
        max_state_history = max(
            max(state_history_offsets),
            max(handoff_state_history_offsets),
        )
        frame_history: deque[np.ndarray] = deque(maxlen=max_visual_history + 1)
        state_history: deque[list[int]] = deque(
            maxlen=max_state_history + 1
        )
        print(
            f"[visual_act] visual history offsets={list(history_offsets)} "
            + (
                f"(2x{len(history_offsets) // 2} mosaic)"
                if len(history_offsets) > 1
                else "(single frame)"
            ),
            flush=True,
        )
        print(
            f"[visual_act] state history offsets={list(state_history_offsets)} "
            f"tactile={tactile_mode} "
            f"(state_dim={JOINT_COUNT * len(state_history_offsets) + (12 if tactile_mode == 'mass-contact' else 0)})",
            flush=True,
        )
        camera = open_camera(args)
        if args.record_rated_attempts:
            recorder = RatedAttemptRecorder(args)
        if args.enable_motion:
            ros.publish_settings(args)
        else:
            print("[visual_act] DRY RUN: predictions only; no settings or commands published", flush=True)
        inference_description = (
            f"temporal ensemble coeff={args.temporal_ensemble_coeff:g}; re-plan every frame"
            if args.temporal_ensemble_coeff is not None
            else f"executes {args.n_action_steps} actions/chunk"
        )
        if args.chunk_boundary_blend_frames:
            inference_description += (
                f"; blends {args.chunk_boundary_blend_frames} boundary frames"
                + (
                    " on thumb only"
                    if args.chunk_boundary_blend_thumb_only
                    else ""
                )
            )
        print(
            f"[visual_act] starts DISARMED; ACT {inference_description}. "
            "Focus the camera window; R resets, SPACE toggles"
            + (
                ", P moves to the second policy and starts it"
                if handoff_controller is not None
                else ""
            )
            + (
                ", 1-8 intervene from the action library, 9 returns to ACT"
                if action_intervention is not None
                else ""
            )
            + ", Q/ESC exits.",
            flush=True,
        )

        armed = False
        resetting = False
        handoff_completed = False
        handoff_status: str | None = None
        active_since: Optional[float] = None
        ema: Optional[np.ndarray] = None
        latest_raw: Optional[np.ndarray] = None
        latest_published_command: Optional[list[int]] = None
        thumb_keyboard_bias = int(args.thumb_bias_initial)
        thumb_side_keyboard_bias = int(args.thumb_side_bias_initial)
        message = "waiting for SPACE"
        last_inference = 0.0
        last_reset_command = 0.0
        last_handoff_command = 0.0
        last_log = 0.0
        period = 1.0 / args.rate
        active_indices = np.asarray([i not in RESERVED_IDX for i in range(JOINT_COUNT)])
        cube_ready_reference = (
            load_cube_ready_profile(args.cube_ready_profile)
            if args.cube_ready_profile is not None
            else None
        )
        cube_ready_live: CubeMarkerPose | None = None
        cube_ready_streak = 0
        cube_ready = False
        cube_ready_text: str | None = None
        cube_contact_since: float | None = None
        cube_contact_ready = False
        thumb_tip_gate = ThumbTipBeforeTurnGate(
            release_threshold=args.thumb_tip_release_threshold,
            confirm_frames=args.thumb_tip_confirm_frames,
            turn_release_q0=args.thumb_turn_release_q0,
            turn_release_q10=args.thumb_turn_release_q10,
            turn_confirm_frames=args.thumb_turn_confirm_frames,
            enabled=args.thumb_tip_before_turn,
        )
        endpoint_stopper: DemoEndpointStopper | None = None
        if args.auto_stop_endpoint_profile is not None:
            endpoint_indices, endpoint_templates = load_demo_endpoint_profile(
                args.auto_stop_endpoint_profile
            )
            endpoint_stopper = DemoEndpointStopper(
                endpoint_templates,
                endpoint_indices,
                tolerance=args.auto_stop_endpoint_tolerance,
                confirm_frames=args.auto_stop_endpoint_confirm_frames,
                min_active_seconds=args.auto_stop_min_active_seconds,
                departure_delta=args.auto_stop_departure_delta,
            )
            print(
                f"[visual_act] endpoint auto-stop enabled: "
                f"templates={len(endpoint_templates)} "
                f"tolerance={args.auto_stop_endpoint_tolerance:g} "
                f"confirm={args.auto_stop_endpoint_confirm_frames} "
                f"min_active={args.auto_stop_min_active_seconds:g}s "
                f"departure={args.auto_stop_departure_delta:g}",
                flush=True,
            )
        if args.thumb_tip_before_turn:
            print(
                "[visual_act] thumb ordering gate enabled: hold q0/q5/q10="
                f"{tuple(int(v) for v in A4_PRETURN_THUMB_ORIENTATION)}, force q15=0, "
                f"release at measured q15<={args.thumb_tip_release_threshold:g} "
                f"for {args.thumb_tip_confirm_frames} frames; keep q15=0 through "
                f"turn until q0<={args.thumb_turn_release_q0:g} and "
                f"q10<={args.thumb_turn_release_q10:g}",
                flush=True,
            )
        if args.cube_ready_profile is not None:
            if cube_ready_reference is None:
                print(
                    "[visual_act] CUBE GATE: no calibration; place cube correctly "
                    "on the fingers and press G while DISARMED",
                    flush=True,
                )
            else:
                print(
                    f"[visual_act] CUBE GATE: loaded {args.cube_ready_profile}",
                    flush=True,
                )

        while rclpy.ok():
            was_armed = armed
            ros.spin(0.0)
            ok, frame = camera.read()
            if not ok or frame is None:
                armed = False
                if action_intervention is not None:
                    action_intervention.stop()
                message = "camera frame failed; DISARMED"
                if recorder is not None:
                    contact_summary = ros.contact_snapshot(
                        time.monotonic(), args.touch_stale_seconds
                    )
                    recorder.stop(message, contact_summary)
                    ros.end_contact_attempt()
                continue
            now = time.monotonic()
            frame_history.append(frame.copy())
            if args.cube_ready_profile is not None:
                cube_ready_live = detect_cube_marker_pose(
                    frame, min_markers=args.cube_ready_min_markers
                )
                if cube_ready_reference is None:
                    cube_ready_streak = 0
                    cube_ready = False
                    cube_ready_text = "CUBE GATE: press G with cube correctly on fingers"
                elif cube_ready_live is None:
                    cube_ready_streak = 0
                    cube_ready = False
                    cube_ready_text = (
                        f"CUBE NOT READY: need >= {args.cube_ready_min_markers} markers"
                    )
                else:
                    matches, center_error, scale_error = cube_pose_matches(
                        cube_ready_live,
                        cube_ready_reference,
                        center_tolerance=args.cube_ready_center_tolerance,
                        scale_tolerance=(
                            None
                            if args.cube_ready_ignore_scale
                            else args.cube_ready_scale_tolerance
                        ),
                    )
                    cube_ready_streak = cube_ready_streak + 1 if matches else 0
                    cube_ready = cube_ready_streak >= args.cube_ready_confirm_frames
                    cube_ready_text = (
                        f"CUBE {'READY' if cube_ready else 'WAIT'} "
                        f"{min(cube_ready_streak, args.cube_ready_confirm_frames)}/"
                        f"{args.cube_ready_confirm_frames} center={center_error:.3f} "
                        f"scale={'ignored' if args.cube_ready_ignore_scale else f'{scale_error:.3f}'}"
                    )
            elif args.cube_contact_gate:
                contact_sample = ros.contact_snapshot(now, args.touch_stale_seconds)
                start_contact_count = finger_mass_contact_count(
                    contact_sample, args.cube_contact_threshold
                )
                contact_condition = (
                    contact_sample["touch_fresh"]
                    and start_contact_count >= args.cube_contact_min_fingers
                )
                if contact_condition:
                    if cube_contact_since is None:
                        cube_contact_since = now
                else:
                    cube_contact_since = None
                contact_held = (
                    max(0.0, now - cube_contact_since)
                    if cube_contact_since is not None
                    else 0.0
                )
                cube_contact_ready = (
                    contact_condition
                    and contact_held >= args.cube_contact_hold_seconds
                )
                cube_ready_text = (
                    f"START CONTACT {'READY' if cube_contact_ready else 'WAIT'} "
                    f"{start_contact_count}/{args.cube_contact_min_fingers} "
                    f">={args.cube_contact_threshold:g}g hold={contact_held:.2f}s"
                )
            state_fresh = ros.state_is_fresh(args.state_stale_seconds)
            if not state_fresh:
                if armed:
                    print(
                        "[visual_act] AUTO DISARM: ROS state stale",
                        flush=True,
                    )
                armed = False
                message = "ROS state stale; DISARMED"
            state = ros.last_state
            if state is not None:
                state_history.append(list(state))

            if resetting and state is not None and now - last_reset_command >= period:
                if not ros.state_is_fresh(args.state_stale_seconds):
                    message = "reset paused: ROS state stale"
                else:
                    reset_delta = max(
                        abs(G20_OPEN_POSE[i] - state[i])
                        for i in range(JOINT_COUNT)
                        if i not in RESERVED_IDX
                    )
                    if reset_delta <= args.reset_tolerance:
                        resetting = False
                        latest_published_command = None
                        policy.reset()
                        chunk_blender.reset()
                        thumb_tip_gate.reset()
                        message = "reset complete; reposition object, then press SPACE"
                        print("[visual_act] RESET complete; ready for next attempt", flush=True)
                    else:
                        reset_command = limit_from_observed_state(
                            G20_OPEN_POSE, state, args.reset_range_step
                        )
                        if args.enable_motion:
                            ros.publish_pose(reset_command)
                        message = (
                            f"resetting to open: remaining {reset_delta} ticks; "
                            f"step <= {args.reset_range_step}"
                        )
                    last_reset_command = now

            if (
                handoff_controller is not None
                and handoff_controller.active
                and state is not None
                and state_fresh
            ):
                transition = handoff_controller.update(state, now)
                if transition == "warmup_started":
                    frame_history.clear()
                    frame_history.append(frame.copy())
                    state_history.clear()
                    state_history.append(list(state))
                    ema = None
                    latest_raw = None
                    latest_published_command = None
                    assert handoff_policy is not None
                    handoff_policy.reset()
                    handoff_status = "WARMING 20K HISTORY"
                    message = (
                        "20K start pose reached; hold target while filling "
                        f"{args.policy_handoff_warmup_seconds:.1f}s history"
                    )
                    print(f"[visual_act] HANDOFF: {message}", flush=True)
                elif transition == "warmup_reset":
                    handoff_status = "WARMUP RESET"
                    message = (
                        f"handoff pose drifted; error={handoff_controller.last_error:.1f}, "
                        "restarting history warmup"
                    )
                elif transition == "moving":
                    handoff_status = "MOVING TO 20K START"
                    message = (
                        f"handoff moving to 20K start; "
                        f"error={handoff_controller.last_error:.1f}/"
                        f"{args.policy_handoff_tolerance:g}"
                    )
                elif transition == "warming":
                    handoff_status = "WARMING 20K HISTORY"
                    assert handoff_controller.warmup_started_at is not None
                    warmed = now - handoff_controller.warmup_started_at
                    message = (
                        f"20K history warmup {warmed:.1f}/"
                        f"{args.policy_handoff_warmup_seconds:.1f}s"
                    )
                elif transition == "timeout":
                    handoff_status = None
                    latest_published_command = None
                    if args.enable_motion:
                        ros.publish_pose(observed_hold_pose(state))
                        ros.publish_settings(args)
                    message = (
                        "handoff timed out before reaching the 20K start pose; "
                        "DISARMED"
                    )
                    print(f"[visual_act] HANDOFF ABORTED: {message}", flush=True)
                elif transition == "complete":
                    assert handoff_policy is not None
                    assert handoff_preprocessor is not None
                    assert handoff_postprocessor is not None
                    policy = handoff_policy
                    preprocessor = handoff_preprocessor
                    postprocessor = handoff_postprocessor
                    history_offsets = handoff_history_offsets
                    state_history_offsets = handoff_state_history_offsets
                    tactile_mode = handoff_tactile_mode
                    handoff_completed = True
                    handoff_status = None
                    ema = None
                    latest_raw = None
                    latest_published_command = None
                    last_inference = 0.0
                    policy.reset()
                    chunk_blender = ChunkBoundaryBlender(
                        args.policy_handoff_n_action_steps,
                        min(
                            args.chunk_boundary_blend_frames,
                            args.policy_handoff_n_action_steps,
                        ),
                        chunk_blend_indices,
                    )
                    thumb_tip_gate.reset()
                    endpoint_stopper = None
                    armed = True
                    active_since = now
                    message = "20K policy ARMED after P-key handoff"
                    print(
                        "[visual_act] HANDOFF COMPLETE: now running "
                        f"{args.policy_handoff_checkpoint_dir}",
                        flush=True,
                    )

                if (
                    handoff_controller.active
                    and now - last_handoff_command >= period
                ):
                    handoff_command = handoff_controller.command(
                        state, args.policy_handoff_range_step
                    )
                    if args.enable_motion:
                        ros.publish_pose(handoff_command)
                    latest_published_command = handoff_command
                    last_handoff_command = now

            waiting_for_rating = bool(recorder and recorder.awaiting_score)
            if (
                action_intervention is not None
                and action_intervention.active
                and armed
                and state is not None
                and state_fresh
                and not waiting_for_rating
                and not resetting
                and not (
                    handoff_controller is not None
                    and handoff_controller.active
                )
                and now - last_inference >= period
                and (
                    tactile_mode == "none"
                    or ros.contact_snapshot(now, args.touch_stale_seconds)[
                        "touch_fresh"
                    ]
                )
            ):
                latest_raw = None
                target = action_intervention.next_target()
                target_delta = float(
                    np.max(
                        np.abs(
                            np.asarray(target, dtype=np.float32)[active_indices]
                            - np.asarray(state, dtype=np.float32)[active_indices]
                        )
                    )
                )
                if target_delta > args.max_target_delta:
                    was_publishing = armed
                    armed = False
                    message = (
                        f"library target delta {target_delta:.1f} too large; "
                        "DISARMED"
                    )
                    if was_publishing:
                        print(
                            f"[visual_act] AUTO DISARM: {message}",
                            flush=True,
                        )
                else:
                    if args.keyboard_thumb_bias:
                        target = apply_thumb_joint_bias(
                            target,
                            args.keyboard_thumb_bias_joint,
                            thumb_keyboard_bias,
                        )
                    if args.keyboard_thumb_side_bias:
                        target = apply_thumb_joint_bias(
                            target,
                            5,
                            thumb_side_keyboard_bias,
                        )
                    command = limit_from_observed_state(
                        target, state, args.max_range_step
                    )
                    if armed and args.enable_motion:
                        ros.publish_pose(command)
                        latest_published_command = command
                    status = action_intervention.status or "inactive"
                    message = (
                        f"LIBRARY {status}; press 9 for ACT; "
                        f"step <= {args.max_range_step}"
                    )
                    if now - last_log >= args.log_period:
                        publishing = armed and args.enable_motion
                        print(
                            f"[visual_act] "
                            f"{'PUBLISH' if publishing else 'PREVIEW ONLY (not published)'} "
                            f"{message} "
                            f"{'cmd' if publishing else 'would_cmd'}={command}",
                            flush=True,
                        )
                        last_log = now
                last_inference = now

            if (
                state is not None
                and state_fresh
                and not waiting_for_rating
                and not resetting
                and not (
                    action_intervention is not None
                    and action_intervention.active
                )
                and not (
                    handoff_controller is not None
                    and handoff_controller.active
                )
                and now - last_inference >= period
            ):
                observation = make_observation(
                    frame_history,
                    state,
                    history_offsets,
                    state_history,
                    state_history_offsets,
                    tactile_mode,
                    ros.last_mass_values,
                    args.contact_on_threshold,
                )
                processed = preprocessor(observation)
                with torch.inference_mode():
                    action = policy.select_action(
                        {key: processed[key] for key in policy.config.input_features}
                    )
                policy_raw = (
                    postprocessor(action)
                    .detach()
                    .cpu()
                    .numpy()
                    .reshape(-1)[:JOINT_COUNT]
                    .astype(np.float32)
                )
                (
                    latest_raw,
                    chunk_blend_active,
                    chunk_blend_weight,
                    chunk_boundary_started,
                ) = chunk_blender.apply(policy_raw, ema)
                if chunk_boundary_started:
                    print(
                        "[visual_act] CHUNK BOUNDARY: cross-fading "
                        f"{args.chunk_boundary_blend_frames} frames"
                        + (
                            " on thumb q0/q5/q10/q15 only"
                            if args.chunk_boundary_blend_thumb_only
                            else ""
                        ),
                        flush=True,
                    )
                if ema is None:
                    ema = latest_raw.copy()
                else:
                    ema = args.ema_alpha * ema + (1.0 - args.ema_alpha) * latest_raw

                raw_bad = bool(
                    np.any(policy_raw[active_indices] < -args.max_raw_overshoot)
                    or np.any(policy_raw[active_indices] > 255.0 + args.max_raw_overshoot)
                )
                target_delta = float(
                    np.max(
                        np.abs(
                            ema[active_indices]
                            - np.asarray(state, dtype=np.float32)[active_indices]
                        )
                    )
                )
                if raw_bad:
                    was_publishing = armed
                    armed = False
                    active_raw = latest_raw[active_indices]
                    message = (
                        f"raw {active_raw.min():.1f}/{active_raw.max():.1f} outside "
                        f"[-{args.max_raw_overshoot:.0f},"
                        f"{255 + args.max_raw_overshoot:.0f}]; DISARMED"
                    )
                    if was_publishing:
                        print(
                            f"[visual_act] AUTO DISARM: {message}",
                            flush=True,
                        )
                elif target_delta > args.max_target_delta:
                    was_publishing = armed
                    armed = False
                    message = f"target delta {target_delta:.1f} too large; DISARMED"
                    if was_publishing:
                        print(
                            f"[visual_act] AUTO DISARM: {message}",
                            flush=True,
                        )
                else:
                    target = clamp_pose(ema)
                    if handoff_completed:
                        thumb_gate_waiting = False
                        thumb_gate_released = False
                    else:
                        target, thumb_gate_waiting, thumb_gate_released = (
                            thumb_tip_gate.apply(target, state)
                        )
                    thumb_push_offset = (
                        0 if handoff_completed else args.thumb_final_push_offset
                    )
                    target, thumb_final_push_active = apply_thumb_final_push_offset(
                        target, state, thumb_push_offset
                    )
                    if args.keyboard_thumb_bias:
                        target = apply_thumb_joint_bias(
                            target,
                            args.keyboard_thumb_bias_joint,
                            thumb_keyboard_bias,
                        )
                    if args.keyboard_thumb_side_bias:
                        target = apply_thumb_joint_bias(
                            target,
                            5,
                            thumb_side_keyboard_bias,
                        )
                    command = limit_from_observed_state(target, state, args.max_range_step)
                    if armed and args.enable_motion:
                        ros.publish_pose(command)
                        latest_published_command = command
                    if thumb_final_push_active:
                        message = (
                            "THUMB FINAL PUSH: q15 target="
                            f"{target[15]} ({thumb_push_offset:+d})"
                        )
                    elif thumb_gate_waiting:
                        if thumb_tip_gate.stage == "tip":
                            message = (
                                "THUMB TIP GATE: q15="
                                f"{thumb_tip_gate.last_tip_position:.0f}>"
                                f"{args.thumb_tip_release_threshold:g}; turn held"
                            )
                        else:
                            message = (
                                "THUMB TURN: q15 held at 0; "
                                f"q0={thumb_tip_gate.last_q0:.0f}/"
                                f"{args.thumb_turn_release_q0:g} "
                                f"q10={thumb_tip_gate.last_q10:.0f}/"
                                f"{args.thumb_turn_release_q10:g}"
                            )
                    elif thumb_gate_released:
                        message = "THUMB TURN ALIGNED: q15 release allowed"
                    elif chunk_blend_active:
                        message = (
                            "THUMB CHUNK BLEND: "
                            f"{chunk_blend_weight:.0%} new plan"
                            if args.chunk_boundary_blend_thumb_only
                            else "CHUNK BLEND: "
                            f"{chunk_blend_weight:.0%} new plan"
                        )
                    else:
                        message = (
                            f"max target delta {target_delta:.1f}; "
                            f"step <= {args.max_range_step}"
                        )
                    if now - last_log >= args.log_period:
                        publishing = armed and args.enable_motion
                        print(
                            f"[visual_act] {'PUBLISH' if publishing else 'PREVIEW ONLY (not published)'} "
                            f"raw=({policy_raw.min():.1f},{policy_raw.max():.1f}) "
                            f"max_delta={target_delta:.1f} "
                            f"{'cmd' if publishing else 'would_cmd'}={command}",
                            flush=True,
                        )
                        last_log = now
                last_inference = now

            if (
                armed
                and recorder is not None
                and latest_published_command is not None
                and state is not None
            ):
                contact_sample = ros.contact_snapshot(now, args.touch_stale_seconds)
                recorder.add(
                    frame,
                    state,
                    latest_published_command,
                    now,
                    ros.last_mass_values,
                    contact_sample["contacts"] if contact_sample["touch_fresh"] else None,
                    contact_sample["touch_age_seconds"],
                    control_source=(
                        f"action_library_{action_intervention.action_id}"
                        if action_intervention is not None
                        and action_intervention.active
                        else "act_policy"
                    ),
                    thumb_tip_bias=thumb_keyboard_bias,
                    thumb_side_bias=thumb_side_keyboard_bias,
                )
                if args.stop_on_contact_success and contact_sample["success_gate_met"]:
                    armed = False
                    active_since = None
                    message = (
                        f"touch success: {contact_sample['finger_contact_count']} fingers "
                        f"for {contact_sample['continuous_contact_seconds']:.1f}s; DISARMED"
                    )
                    print(f"[visual_act] {message}", flush=True)

            if (
                armed
                and endpoint_stopper is not None
                and state is not None
                and endpoint_stopper.update(state, now)
            ):
                if args.enable_motion and args.hold_on_disarm:
                    hold_pose = observed_hold_pose(state)
                    ros.publish_pose(hold_pose)
                    latest_published_command = hold_pose
                armed = False
                active_since = None
                message = (
                    "demo endpoint reached: "
                    f"error={endpoint_stopper.nearest_error:.1f} "
                    f"for {endpoint_stopper.confirmed} frames; DISARMED"
                )
                print(f"[visual_act] AUTO STOP: {message}", flush=True)

            if (
                armed
                and args.max_active_seconds > 0
                and active_since is not None
                and now - active_since >= args.max_active_seconds
            ):
                armed = False
                active_since = None
                message = f"{args.max_active_seconds:.1f}s limit reached; DISARMED"
                print(f"[visual_act] {message}", flush=True)

            if was_armed and not armed:
                if action_intervention is not None:
                    action_intervention.stop()
                if recorder is not None:
                    print(f"[visual_act] DISARM REASON: {message}", flush=True)
                    contact_summary = ros.contact_snapshot(
                        now, args.touch_stale_seconds
                    )
                    recorder.stop(message, contact_summary)
                    ros.end_contact_attempt()

            display = frame.copy()
            live_contact = ros.contact_snapshot(now, args.touch_stale_seconds)
            shown_contact = (
                recorder.contact_summary
                if recorder is not None
                and recorder.awaiting_score
                and recorder.contact_summary is not None
                else live_contact
            )
            auxiliary_status = cube_ready_text
            if endpoint_stopper is not None:
                endpoint_error = (
                    f"{endpoint_stopper.nearest_error:.1f}"
                    if np.isfinite(endpoint_stopper.nearest_error)
                    else "--"
                )
                endpoint_text = (
                    f"END {'TRACK' if armed else 'READY'} "
                    f"depart={'Y' if endpoint_stopper.departed else 'N'} "
                    f"error={endpoint_error}/{args.auto_stop_endpoint_tolerance:g} "
                    f"settle={endpoint_stopper.confirmed}/"
                    f"{args.auto_stop_endpoint_confirm_frames}"
                )
                auxiliary_status = (
                    endpoint_text
                    if auxiliary_status is None
                    else f"{auxiliary_status} | {endpoint_text}"
                )
            draw_status(
                display,
                armed,
                args.enable_motion,
                latest_raw,
                message,
                awaiting_score=bool(recorder and recorder.awaiting_score),
                resetting=resetting,
                contact_summary=None if args.ignore_touch else shown_contact,
                cube_ready_text=auxiliary_status,
                minimal_overlay=args.minimal_overlay,
                handoff_enabled=handoff_controller is not None,
                handoff_status=handoff_status,
                keyboard_thumb_bias_joint=(
                    args.keyboard_thumb_bias_joint
                    if args.keyboard_thumb_bias
                    else None
                ),
                keyboard_thumb_bias=thumb_keyboard_bias,
                keyboard_thumb_side_bias=(
                    thumb_side_keyboard_bias
                    if args.keyboard_thumb_side_bias
                    else None
                ),
                action_intervention_enabled=action_intervention is not None,
                action_intervention_status=(
                    action_intervention.status
                    if action_intervention is not None
                    else None
                ),
            )
            cv2.imshow("G20 visual ACT - first bring-up", display)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                handoff_is_active = bool(
                    handoff_controller is not None
                    and handoff_controller.active
                )
                if (
                    args.enable_motion
                    and state is not None
                    and (handoff_is_active or (armed and args.hold_on_disarm))
                ):
                    ros.publish_pose(observed_hold_pose(state))
                if armed and recorder is not None:
                    recorder.stop(
                        "operator exit",
                        ros.contact_snapshot(now, args.touch_stale_seconds),
                    )
                    ros.end_contact_attempt()
                break
            if args.keyboard_thumb_bias and key in (
                ord("["),
                ord("]"),
                ord("\\"),
            ):
                if key == ord("["):
                    thumb_keyboard_bias = adjust_bounded_keyboard_bias(
                        thumb_keyboard_bias,
                        -args.thumb_bias_step,
                        args.thumb_bias_limit,
                    )
                elif key == ord("]"):
                    thumb_keyboard_bias = adjust_bounded_keyboard_bias(
                        thumb_keyboard_bias,
                        args.thumb_bias_step,
                        args.thumb_bias_limit,
                    )
                else:
                    thumb_keyboard_bias = 0
                message = (
                    f"keyboard thumb bias q{args.keyboard_thumb_bias_joint}="
                    f"{thumb_keyboard_bias:+d}"
                )
                print(f"[visual_act] {message}", flush=True)
                continue
            if args.keyboard_thumb_side_bias and key in (
                ord("a"),
                ord("A"),
                ord("d"),
                ord("D"),
                ord("s"),
                ord("S"),
            ):
                if key in (ord("a"), ord("A")):
                    thumb_side_keyboard_bias = adjust_bounded_keyboard_bias(
                        thumb_side_keyboard_bias,
                        -args.thumb_side_bias_step,
                        args.thumb_side_bias_limit,
                    )
                elif key in (ord("d"), ord("D")):
                    thumb_side_keyboard_bias = adjust_bounded_keyboard_bias(
                        thumb_side_keyboard_bias,
                        args.thumb_side_bias_step,
                        args.thumb_side_bias_limit,
                    )
                else:
                    thumb_side_keyboard_bias = 0
                message = (
                    "keyboard thumb side bias q5="
                    f"{thumb_side_keyboard_bias:+d}"
                )
                print(f"[visual_act] {message}", flush=True)
                continue
            if recorder is not None and recorder.awaiting_score:
                score_by_key = {ord("0"): 0.0, ord("5"): 0.5, ord("1"): 1.0}
                if key in score_by_key:
                    human_score = score_by_key[key]
                    score = human_score
                    contact_passed = bool(
                        recorder.contact_summary
                        and recorder.contact_summary.get("success_gate_met")
                    )
                    if (
                        human_score == 1.0
                        and args.require_touch_for_score_one
                        and not contact_passed
                    ):
                        score = 0.5
                        print(
                            "[visual_act] human score=1 downgraded to 0.5: "
                            f"need thumb + >= {args.min_contact_fingers - 1} other "
                            f"fingers for {args.contact_hold_seconds:.1f}s with fresh touch",
                            flush=True,
                        )
                    recorder.rate(score, human_score=human_score)
                    if args.reset_after_rating and args.enable_motion:
                        resetting = True
                        last_reset_command = 0.0
                        latest_raw = None
                        message = (
                            f"saved score {score:g} (human {human_score:g}); "
                            "resetting to open pose"
                        )
                        print(
                            f"[visual_act] score={score:g}; RESETTING to G20 open pose",
                            flush=True,
                        )
                    else:
                        message = f"saved human score {score:g}; ready for next attempt"
                elif key == ord(" "):
                    message = "rate the previous attempt with 0, 5, or 1 first"
                continue
            if resetting:
                if key == ord(" "):
                    message = "reset in progress; wait for RESET complete"
                continue
            if key in (ord("p"), ord("P")):
                if handoff_controller is None:
                    message = (
                        "P handoff unavailable; supply the handoff checkpoint "
                        "and start pose"
                    )
                elif handoff_completed:
                    message = "already running the 20K handoff policy"
                elif handoff_controller.active:
                    message = "policy handoff already in progress; SPACE aborts"
                elif not args.enable_motion:
                    message = "P handoff requires --enable-motion"
                elif not state_fresh or state is None:
                    message = "cannot start policy handoff: ROS state is stale"
                else:
                    armed = False
                    active_since = None
                    if action_intervention is not None:
                        action_intervention.stop()
                    ema = None
                    latest_raw = None
                    latest_published_command = None
                    last_handoff_command = 0.0
                    policy.reset()
                    chunk_blender.reset()
                    thumb_tip_gate.reset()
                    handoff_controller.begin(now)
                    handoff_status = "MOVING TO 20K START"
                    handoff_current = (
                        args.policy_handoff_current_limit
                        if args.policy_handoff_current_limit is not None
                        else args.current_limit
                    )
                    handoff_speed = (
                        args.policy_handoff_speed_limit
                        if args.policy_handoff_speed_limit is not None
                        else args.speed_limit
                    )
                    ros.publish_limits(
                        args.side,
                        handoff_current,
                        handoff_speed,
                    )
                    message = "P pressed: moving to the 20K policy start pose"
                    print(
                        "[visual_act] HANDOFF START: primary policy disarmed; "
                        f"moving with step<={args.policy_handoff_range_step}, "
                        f"current={handoff_current}, speed={handoff_speed}",
                        flush=True,
                    )
                continue
            if handoff_controller is not None and handoff_controller.active:
                if key == ord(" "):
                    handoff_controller.abort()
                    handoff_status = None
                    armed = False
                    active_since = None
                    latest_published_command = None
                    if args.enable_motion and state is not None:
                        ros.publish_pose(observed_hold_pose(state))
                        ros.publish_settings(args)
                    message = "operator aborted policy handoff; primary remains DISARMED"
                    print(f"[visual_act] HANDOFF ABORTED: {message}", flush=True)
                elif key != 255:
                    message = "handoff in progress; SPACE aborts, Q/ESC exits"
                continue
            if action_intervention is not None and key in tuple(
                ord(str(value)) for value in range(1, 10)
            ):
                if not armed:
                    message = (
                        "numbered intervention requires ARMED policy control; "
                        "press SPACE first"
                    )
                elif not state_fresh or state is None:
                    message = "cannot intervene: ROS state is stale"
                elif key == ord("9"):
                    if action_intervention.active:
                        previous_id = action_intervention.action_id
                        action_intervention.stop()
                        ema = None
                        latest_raw = None
                        latest_published_command = None
                        last_inference = 0.0
                        policy.reset()
                        chunk_blender.reset()
                        thumb_tip_gate.reset()
                        message = (
                            f"A{previous_id} intervention stopped; "
                            "ACT policy resumed with current history"
                        )
                        print(
                            f"[visual_act] POLICY RESUME: {message}",
                            flush=True,
                        )
                    else:
                        message = "ACT policy already controls the hand"
                else:
                    action_id = int(chr(key))
                    if action_id not in action_intervention.library.primitives:
                        available = sorted(
                            value
                            for value in action_intervention.library.primitives
                            if 1 <= value <= 8
                        )
                        message = (
                            f"action {action_id} is not in this library; "
                            f"available keys={available}"
                        )
                    else:
                        nearest, error, playback_frames = (
                            action_intervention.begin(action_id, list(state))
                        )
                        ema = None
                        latest_raw = None
                        latest_published_command = None
                        last_inference = 0.0
                        policy.reset()
                        chunk_blender.reset()
                        thumb_tip_gate.reset()
                        primitive = action_intervention.library.primitives[action_id]
                        message = (
                            f"LIBRARY A{action_id} {primitive.name}: "
                            f"nearest source frame {nearest + 1}/"
                            f"{len(primitive.trajectory)}, error={error:.1f}, "
                            f"playback={playback_frames} frames; press 9 for ACT"
                        )
                        print(
                            f"[visual_act] ACTION INTERVENTION: {message}",
                            flush=True,
                        )
                continue
            if key in (ord("r"), ord("R")):
                if not args.enable_motion:
                    message = "reset unavailable in dry-run; restart with --enable-motion"
                elif armed:
                    message = "press SPACE to DISARM before resetting"
                elif not ros.state_is_fresh(args.state_stale_seconds):
                    message = "cannot reset: ROS state is stale"
                else:
                    resetting = True
                    if action_intervention is not None:
                        action_intervention.stop()
                    last_reset_command = 0.0
                    frame_history.clear()
                    frame_history.append(frame.copy())
                    state_history.clear()
                    state_history.append(list(state))
                    ema = None
                    latest_raw = None
                    latest_published_command = None
                    policy.reset()
                    chunk_blender.reset()
                    thumb_tip_gate.reset()
                    message = "operator requested reset to G20 open pose"
                    print(
                        "[visual_act] RESETTING to G20 open pose; "
                        "wait for RESET complete",
                        flush=True,
                    )
                continue
            if key in (ord("g"), ord("G")):
                if args.cube_ready_profile is None:
                    message = "restart with --cube-ready-profile to enable cube gate"
                elif armed:
                    message = "DISARM before calibrating cube-ready pose"
                elif cube_ready_live is None:
                    message = (
                        f"calibration failed: need >= {args.cube_ready_min_markers} "
                        "markers on one cube face"
                    )
                else:
                    save_cube_ready_profile(args.cube_ready_profile, cube_ready_live)
                    cube_ready_reference = cube_ready_live
                    cube_ready_streak = 1
                    cube_ready = args.cube_ready_confirm_frames <= 1
                    message = (
                        "cube-ready pose calibrated; hold still until CUBE READY, "
                        "then press SPACE"
                    )
                    print(
                        f"[visual_act] CUBE GATE calibrated: "
                        f"{args.cube_ready_profile} pose={asdict(cube_ready_live)}",
                        flush=True,
                    )
                continue
            if key == ord(" "):
                if armed:
                    if args.enable_motion and args.hold_on_disarm and state is not None:
                        ros.publish_pose(observed_hold_pose(state))
                    armed = False
                    active_since = None
                    if action_intervention is not None:
                        action_intervention.stop()
                    message = "operator DISARMED"
                    if recorder is not None:
                        contact_summary = ros.contact_snapshot(
                            now, args.touch_stale_seconds
                        )
                        recorder.stop(message, contact_summary)
                        ros.end_contact_attempt()
                    print("[visual_act] DISARMED; command publishing stopped", flush=True)
                elif (
                    args.cube_ready_profile is not None
                    and cube_ready_reference is None
                ):
                    message = "cannot arm: calibrate cube-ready pose with G first"
                    print(f"[visual_act] ARM REFUSED: {message}", flush=True)
                elif args.cube_ready_profile is not None and not cube_ready:
                    message = "cannot arm: cube is not stably in the calibrated finger pose"
                    print(f"[visual_act] ARM REFUSED: {message}", flush=True)
                elif args.cube_contact_gate and not cube_contact_ready:
                    message = "cannot arm: no stable cube contact on the fingers"
                    print(f"[visual_act] ARM REFUSED: {message}", flush=True)
                elif ros.state_is_fresh(args.state_stale_seconds):
                    armed = True
                    active_since = now
                    if action_intervention is not None:
                        action_intervention.stop()
                    frame_history.clear()
                    frame_history.append(frame.copy())
                    state_history.clear()
                    state_history.append(list(state))
                    ema = None
                    latest_published_command = None
                    policy.reset()
                    chunk_blender.reset()
                    thumb_tip_gate.reset()
                    if endpoint_stopper is not None:
                        endpoint_stopper.reset(state, now)
                    if recorder is not None:
                        recorder.start()
                        ros.begin_contact_attempt(now)
                    message = "operator ARMED"
                    print(
                        "[visual_act] ARMED"
                        + ("; publishing limited commands" if args.enable_motion else "; dry-run only"),
                        flush=True,
                    )
                else:
                    message = "cannot arm: ROS state is stale"
        return 0
    except RuntimeError as exc:
        print(f"[visual_act] error: {exc}", file=sys.stderr)
        return 3
    finally:
        if camera is not None:
            camera.release()
        if recorder is not None:
            recorder.close()
        cv2.destroyAllWindows()
        if ros is not None:
            ros.close()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
