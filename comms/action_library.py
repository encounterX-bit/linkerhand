"""MediaPipe motion primitives and G20 SDK-range trajectory execution.

This module is hardware-free.  It owns the on-disk action-library contract,
landmark feature extraction, streaming template matching, and trajectory queue.
The ROS/camera CLI lives in :mod:`src.comms.action_library_teleop`.
"""
from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np


JOINT_COUNT = 20
RESERVED_IDX = (11, 12, 13, 14)
ACTIVE_IDX = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 16, 17, 18, 19)
THUMB_IDX = (0, 5, 10, 15)
G20_OPEN_POSE = np.asarray([
    255, 255, 255, 255, 255,
    255, 193, 148, 105, 42,
    245, 255, 255, 255, 255,
    255, 255, 255, 255, 255,
], dtype=np.float32)
G20_ACTION16_TO_CMD20 = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 16, 17, 18, 19)
G20_SIM_LOWER16 = np.asarray([
    0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, -0.17, -0.17, -0.17, -0.17,
    0.0,
    0.0, 0.0, 0.0, 0.0, 0.0,
], dtype=np.float32)
G20_SIM_UPPER16 = np.asarray([
    0.79, 1.4, 1.4, 1.4, 1.4,
    1.4, 0.17, 0.17, 0.17, 0.17,
    1.22,
    1.05, 1.57, 1.57, 1.57, 1.57,
], dtype=np.float32)


def thumb_roundtrip_trajectory(
    source_trajectory: np.ndarray,
    current_pose: Sequence[float],
) -> tuple[np.ndarray, int, float]:
    """Build a thumb-only nearest-frame -> endpoint -> frame-0 trajectory.

    The nearest source frame is selected using only q0/q5/q10/q15. Every
    non-thumb active channel is held at ``current_pose`` for the entire result.
    """
    source = np.asarray(source_trajectory, dtype=np.float32)
    current = np.asarray(current_pose, dtype=np.float32).reshape(-1)
    if source.ndim != 2 or source.shape[1] != JOINT_COUNT or not len(source):
        raise ValueError(
            f"source trajectory must be non-empty (T,20), got {source.shape}"
        )
    if current.shape != (JOINT_COUNT,):
        raise ValueError("current pose must contain 20 values")
    if not np.all(np.isfinite(source)) or not np.all(np.isfinite(current)):
        raise ValueError("source trajectory and current pose must be finite")

    thumb = list(THUMB_IDX)
    differences = source[:, thumb] - current[thumb]
    errors = np.sqrt(np.mean(np.square(differences), axis=1))
    nearest = int(np.argmin(errors))

    outward = source[nearest:]
    returning = source[-2::-1]
    thumb_path = (
        np.concatenate((outward, returning), axis=0)
        if len(returning)
        else outward.copy()
    )
    result = np.repeat(current[None, :], len(thumb_path), axis=0)
    result[:, thumb] = thumb_path[:, thumb]
    result[:, list(RESERVED_IDX)] = 255.0
    return result, nearest, float(errors[nearest])

# MediaPipe bone edges. Unit directions make matching insensitive to hand size.
_BONES = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
)
_FINGER_CHAINS = (
    (1, 2, 3, 4),
    (0, 5, 6, 7, 8),
    (0, 9, 10, 11, 12),
    (0, 13, 14, 15, 16),
    (0, 17, 18, 19, 20),
)
_TIPS = (4, 8, 12, 16, 20)
FEATURE_PROFILE_FULL = "full_3d_v1"
FEATURE_PROFILE_NO_FINGER_SPLAY = "finger_flexion_no_splay_v1"
FEATURE_PROFILE_THUMB_LITTLE_CONTACT = "finger_flexion_thumb_little_contact_v2"
FEATURE_PROFILES = (
    FEATURE_PROFILE_FULL,
    FEATURE_PROFILE_NO_FINGER_SPLAY,
    FEATURE_PROFILE_THUMB_LITTLE_CONTACT,
)
PHASE_MAPPING_LINEAR = "frame_linear_v1"
PHASE_MAPPING_MOTION = "motion_progress_v1"
PHASE_MAPPING_THUMB_LITTLE_CONTACT = "thumb_little_contact_v1"
PHASE_MAPPING_THUMB_LITTLE_ROUNDTRIP = "thumb_little_roundtrip_v1"
PHASE_MAPPINGS = (
    PHASE_MAPPING_LINEAR,
    PHASE_MAPPING_MOTION,
    PHASE_MAPPING_THUMB_LITTLE_CONTACT,
    PHASE_MAPPING_THUMB_LITTLE_ROUNDTRIP,
)
THUMB_LITTLE_DISTANCE_FEATURE_INDEX = 77


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 1e-8 else np.zeros(3, dtype=np.float32)


def _without_lateral(vector: np.ndarray) -> np.ndarray:
    """Project a hand-base vector onto the x/z finger-flexion plane."""
    projected = np.asarray(vector, dtype=np.float32).copy()
    projected[1] = 0.0
    return projected


def landmark_feature(
    landmarks: Sequence[Sequence[float]],
    *,
    feature_profile: str = FEATURE_PROFILE_FULL,
) -> np.ndarray:
    """Convert hand-base ``(21,3)`` landmarks into an 83-D matching feature.

    The feature contains bone unit directions, internal bend angles, normalized
    thumb-to-fingertip distances, and normalized wrist-to-tip distances. It is
    invariant to translation and uniform hand scale. The caller is expected to
    use the repo's hand-base transform so camera rotation is removed as well.

    ``finger_flexion_no_splay_v1`` projects index-through-little-finger vectors
    onto the hand-base x/z flexion plane. This deliberately removes their
    lateral (+/-y) spread while retaining full 3-D thumb geometry.

    ``finger_flexion_thumb_little_contact_v2`` behaves the same way except that
    feature 77 keeps the full 3-D normalized thumb-tip to little-finger-tip
    distance. This preserves one contact signal without reintroducing the four
    finger spread directions. All profiles remain 83-D.
    """
    if feature_profile not in FEATURE_PROFILES:
        raise ValueError(f"unsupported landmark feature profile {feature_profile!r}")
    ignore_finger_splay = feature_profile in (
        FEATURE_PROFILE_NO_FINGER_SPLAY,
        FEATURE_PROFILE_THUMB_LITTLE_CONTACT,
    )
    keep_thumb_little_contact = (
        feature_profile == FEATURE_PROFILE_THUMB_LITTLE_CONTACT
    )
    points = np.asarray(landmarks, dtype=np.float32)
    if points.shape != (21, 3) or not np.all(np.isfinite(points)):
        raise ValueError(f"landmarks must be finite (21,3), got {points.shape}")
    points = points - points[0]
    palm_scale = float(np.linalg.norm(points[5] - points[17]))
    if palm_scale <= 1e-8:
        raise ValueError("degenerate hand: index-to-little MCP distance is zero")

    values: list[float] = []
    for bone_index, (start, end) in enumerate(_BONES):
        direction = points[end] - points[start]
        if ignore_finger_splay and bone_index >= 4:
            direction = _without_lateral(direction)
        values.extend(_unit(direction).tolist())

    for chain_index, chain in enumerate(_FINGER_CHAINS):
        for left, center, right in zip(chain, chain[1:], chain[2:]):
            a = points[left] - points[center]
            b = points[right] - points[center]
            if ignore_finger_splay and chain_index >= 1:
                a = _without_lateral(a)
                b = _without_lateral(b)
            a = _unit(a)
            b = _unit(b)
            values.append(float(np.arccos(np.clip(float(a @ b), -1.0, 1.0))) / np.pi)

    for tip in _TIPS[1:]:
        delta = points[tip] - points[_TIPS[0]]
        if ignore_finger_splay and not (
            keep_thumb_little_contact and tip == _TIPS[-1]
        ):
            delta = _without_lateral(delta)
        values.append(float(np.linalg.norm(delta)) / palm_scale)
    for tip_index, tip in enumerate(_TIPS):
        delta = points[tip]
        if ignore_finger_splay and tip_index >= 1:
            delta = _without_lateral(delta)
        values.append(float(np.linalg.norm(delta)) / palm_scale)
    return np.asarray(values, dtype=np.float32)


def g20_range_to_sim_radians(
    pose: Sequence[float], *, roll_range_ticks: float = 100.0
) -> np.ndarray:
    """Approximate inverse of the default G20 range map for GUI visualization.

    This is deliberately preview-only: the G20 SDK range calibration is not an
    L20 URDF authority.  Exported waypoints always retain the exact SDK values.
    """
    command = _validated_pose(pose)
    active_open = G20_OPEN_POSE[list(G20_ACTION16_TO_CMD20)]
    active = command[list(G20_ACTION16_TO_CMD20)]
    sim = G20_SIM_LOWER16 + (
        (active_open - active) / np.maximum(active_open, 1.0)
    ) * (G20_SIM_UPPER16 - G20_SIM_LOWER16)
    roll_ticks = max(1.0, float(roll_range_ticks))
    for action_index in (6, 7, 8, 9):
        sim[action_index] = (
            (active[action_index] - active_open[action_index])
            / roll_ticks
            * G20_SIM_UPPER16[action_index]
        )
    sim = np.clip(sim, G20_SIM_LOWER16, G20_SIM_UPPER16)
    out = np.zeros(JOINT_COUNT, dtype=np.float32)
    for value, command_index in zip(sim, G20_ACTION16_TO_CMD20):
        out[command_index] = value
    return out


def dtw_distance(left: np.ndarray, right: np.ndarray, band_ratio: float = 0.35) -> float:
    """Return length-normalized dynamic-time-warping distance."""
    a = np.asarray(left, dtype=np.float32)
    b = np.asarray(right, dtype=np.float32)
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[1] or not len(a) or not len(b):
        raise ValueError(f"DTW inputs must be non-empty (T,F) with equal F: {a.shape}, {b.shape}")
    costs = np.sqrt(np.mean((a[:, None, :] - b[None, :, :]) ** 2, axis=2))
    n, m = costs.shape
    band = max(abs(n - m), int(np.ceil(max(n, m) * max(0.0, band_ratio))))
    prev = np.full(m + 1, np.inf, dtype=np.float64)
    prev[0] = 0.0
    for i in range(1, n + 1):
        curr = np.full(m + 1, np.inf, dtype=np.float64)
        lo = max(1, i - band)
        hi = min(m, i + band)
        for j in range(lo, hi + 1):
            curr[j] = float(costs[i - 1, j - 1]) + min(curr[j - 1], prev[j], prev[j - 1])
        prev = curr
    return float(prev[m] / max(n, m))


def interpolate_waypoints(waypoints: Sequence[dict], fps: float = 30.0) -> np.ndarray:
    """Expand ``[{pose:[20], duration:seconds}, ...]`` into a 30 Hz trajectory.

    A waypoint's duration is the travel time from the previous waypoint.  The
    first waypoint is emitted once; its duration is ignored.
    """
    if fps <= 0 or not waypoints:
        raise ValueError("fps must be positive and at least one waypoint is required")
    poses = [_validated_pose(item.get("pose")) for item in waypoints]
    frames = [poses[0]]
    for item, start, end in zip(waypoints[1:], poses, poses[1:]):
        duration = float(item.get("duration", 0.5))
        if duration <= 0:
            raise ValueError("waypoint duration must be positive")
        count = max(1, int(round(duration * fps)))
        for alpha in np.linspace(1.0 / count, 1.0, count):
            frames.append((1.0 - alpha) * start + alpha * end)
    out = np.asarray(frames, dtype=np.float32)
    out[:, RESERVED_IDX] = 255.0
    return out


def _validated_pose(pose: object) -> np.ndarray:
    values = np.asarray(pose, dtype=np.float32).reshape(-1)
    if values.shape != (JOINT_COUNT,) or not np.all(np.isfinite(values)):
        raise ValueError("each robot pose must contain 20 finite values")
    if np.any(values < 0.0) or np.any(values > 255.0):
        raise ValueError("robot pose values must lie in SDK range 0..255")
    values = values.copy()
    values[list(RESERVED_IDX)] = 255.0
    return values


@dataclass(frozen=True)
class Primitive:
    id: int
    name: str
    trajectory: np.ndarray
    templates: tuple[np.ndarray, ...]
    threshold: float = 0.18
    interruptible: bool = False
    cooldown_frames: int = 10
    phase_mapping: str = PHASE_MAPPING_LINEAR
    phase_motion_epsilon: float = 0.0
    phase_endpoint_snap_distance: float = 0.0
    phase_endpoint_window: int = 5
    phase_contact_fraction: float = 0.5
    manual_from_start: bool = False
    best_effort_spread_feedback: bool = False
    max_command_lead: Optional[float] = None


class ActionLibrary:
    """Validated action-library directory."""

    SCHEMA = "linkerhand_g20_action_library_v1"

    def __init__(
        self,
        root: Path,
        primitives: Iterable[Primitive],
        fps: float = 30.0,
        feature_profile: str = FEATURE_PROFILE_FULL,
    ):
        self.root = Path(root)
        self.fps = float(fps)
        if feature_profile not in FEATURE_PROFILES:
            raise ValueError(f"unsupported landmark feature profile {feature_profile!r}")
        self.feature_profile = feature_profile
        self.primitives = {item.id: item for item in primitives}
        if len(self.primitives) == 0:
            raise ValueError("action library contains no primitives")

    @classmethod
    def load(cls, root: Path) -> "ActionLibrary":
        root = Path(root)
        payload = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        if payload.get("schema") != cls.SCHEMA:
            raise ValueError(f"unsupported action library schema {payload.get('schema')!r}")
        fps = float(payload.get("fps", 30.0))
        feature_profile = str(payload.get("feature_profile", FEATURE_PROFILE_FULL))
        if feature_profile not in FEATURE_PROFILES:
            raise ValueError(f"unsupported landmark feature profile {feature_profile!r}")
        items: list[Primitive] = []
        seen: set[int] = set()
        for record in payload.get("primitives", []):
            primitive_id = int(record["id"])
            if primitive_id in seen:
                raise ValueError(f"duplicate primitive id {primitive_id}")
            seen.add(primitive_id)
            trajectory = np.load(root / record["robot_trajectory"], allow_pickle=False)
            trajectory = _validate_trajectory(trajectory)
            templates = tuple(
                _validate_template(
                    np.load(root / path, allow_pickle=False),
                    feature_profile=feature_profile,
                )
                for path in record.get("human_templates", [])
            )
            if not templates:
                raise ValueError(f"primitive {primitive_id} has no human templates")
            phase_mapping = str(
                record.get("phase_mapping", PHASE_MAPPING_LINEAR)
            )
            if phase_mapping not in PHASE_MAPPINGS:
                raise ValueError(
                    f"primitive {primitive_id} has unsupported phase mapping "
                    f"{phase_mapping!r}"
                )
            phase_motion_epsilon = float(record.get("phase_motion_epsilon", 0.0))
            phase_endpoint_snap_distance = float(
                record.get("phase_endpoint_snap_distance", 0.0)
            )
            phase_endpoint_window = int(record.get("phase_endpoint_window", 5))
            phase_contact_fraction = float(
                record.get("phase_contact_fraction", 0.5)
            )
            if (
                not np.isfinite(phase_motion_epsilon)
                or phase_motion_epsilon < 0.0
                or not np.isfinite(phase_endpoint_snap_distance)
                or phase_endpoint_snap_distance < 0.0
                or phase_endpoint_window <= 0
                or not np.isfinite(phase_contact_fraction)
                or not 0.0 < phase_contact_fraction < 1.0
            ):
                raise ValueError(f"primitive {primitive_id} has invalid phase mapping settings")
            max_command_lead_raw = record.get("max_command_lead")
            max_command_lead = (
                None
                if max_command_lead_raw is None
                else float(max_command_lead_raw)
            )
            if max_command_lead is not None and (
                not np.isfinite(max_command_lead)
                or max_command_lead <= 0.0
            ):
                raise ValueError(
                    f"primitive {primitive_id} has invalid max_command_lead"
                )
            items.append(Primitive(
                id=primitive_id,
                name=str(record["name"]),
                trajectory=trajectory,
                templates=templates,
                threshold=float(record.get("threshold", 0.18)),
                interruptible=bool(record.get("interruptible", False)),
                cooldown_frames=max(0, int(record.get("cooldown_frames", 10))),
                phase_mapping=phase_mapping,
                phase_motion_epsilon=phase_motion_epsilon,
                phase_endpoint_snap_distance=phase_endpoint_snap_distance,
                phase_endpoint_window=phase_endpoint_window,
                phase_contact_fraction=phase_contact_fraction,
                manual_from_start=bool(record.get("manual_from_start", False)),
                best_effort_spread_feedback=bool(
                    record.get("best_effort_spread_feedback", False)
                ),
                max_command_lead=max_command_lead,
            ))
        return cls(root, items, fps=fps, feature_profile=feature_profile)

    def feature(self, landmarks: Sequence[Sequence[float]]) -> np.ndarray:
        """Extract a live feature using the profile that loaded the templates."""
        return landmark_feature(landmarks, feature_profile=self.feature_profile)


def _validate_trajectory(value: np.ndarray) -> np.ndarray:
    trajectory = np.asarray(value, dtype=np.float32)
    if trajectory.ndim != 2 or trajectory.shape[1] != JOINT_COUNT or not len(trajectory):
        raise ValueError(f"robot trajectory must be non-empty (T,20), got {trajectory.shape}")
    if not np.all(np.isfinite(trajectory)) or np.any(trajectory < 0) or np.any(trajectory > 255):
        raise ValueError("robot trajectory must contain finite SDK values in 0..255")
    trajectory = trajectory.copy()
    trajectory[:, RESERVED_IDX] = 255.0
    return trajectory


def _validate_template(
    value: np.ndarray,
    *,
    feature_profile: str = FEATURE_PROFILE_FULL,
) -> np.ndarray:
    template = np.asarray(value, dtype=np.float32)
    if template.ndim == 3 and template.shape[1:] == (21, 3):
        template = np.stack([
            landmark_feature(frame, feature_profile=feature_profile)
            for frame in template
        ])
    if template.ndim != 2 or not len(template) or not np.all(np.isfinite(template)):
        raise ValueError(f"human template must be finite (T,F), got {template.shape}")
    return template


def template_phase_axis(
    template: np.ndarray,
    *,
    mapping: str = PHASE_MAPPING_LINEAR,
    motion_epsilon: float = 0.0,
    endpoint_snap_distance: float = 0.0,
    endpoint_window: int = 5,
    contact_fraction: float = 0.5,
) -> np.ndarray:
    """Map template frames to normalized motion phase.

    Linear mapping preserves the original frame-time behavior. Motion mapping
    accumulates only feature displacement above ``motion_epsilon``, so a held
    endpoint does not consume a large part of the action phase. If a stable
    suffix lies within ``endpoint_snap_distance`` of its robust endpoint, that
    complete suffix maps to phase 1.0.
    """
    values = np.asarray(template, dtype=np.float32)
    if values.ndim != 2 or not len(values) or not np.all(np.isfinite(values)):
        raise ValueError("phase template must be finite (T,F)")
    if mapping not in PHASE_MAPPINGS:
        raise ValueError(f"unsupported phase mapping {mapping!r}")
    epsilon = float(motion_epsilon)
    snap = float(endpoint_snap_distance)
    window = int(endpoint_window)
    contact_phase = float(contact_fraction)
    if (
        not np.isfinite(epsilon)
        or epsilon < 0.0
        or not np.isfinite(snap)
        or snap < 0.0
        or window <= 0
        or not np.isfinite(contact_phase)
        or not 0.0 < contact_phase < 1.0
    ):
        raise ValueError("invalid phase mapping settings")

    if mapping == PHASE_MAPPING_LINEAR or len(values) == 1:
        phases = np.linspace(0.0, 1.0, len(values), dtype=np.float32)
    elif mapping == PHASE_MAPPING_THUMB_LITTLE_CONTACT:
        if values.shape[1] <= THUMB_LITTLE_DISTANCE_FEATURE_INDEX:
            raise ValueError(
                "thumb-little contact phase requires feature index "
                f"{THUMB_LITTLE_DISTANCE_FEATURE_INDEX}, got width {values.shape[1]}"
            )
        distance = values[:, THUMB_LITTLE_DISTANCE_FEATURE_INDEX]
        contact_count = min(window, len(distance))
        contact = float(np.max(np.partition(
            distance, contact_count - 1
        )[:contact_count]))
        start = float(distance[0])
        span = start - contact
        if span <= max(1e-6, epsilon):
            phases = np.linspace(0.0, 1.0, len(values), dtype=np.float32)
        else:
            phases = np.clip((start - distance) / span, 0.0, 1.0)
            phases = np.maximum.accumulate(phases).astype(np.float32)
            phases[0] = 0.0
    elif mapping == PHASE_MAPPING_THUMB_LITTLE_ROUNDTRIP:
        if values.shape[1] <= THUMB_LITTLE_DISTANCE_FEATURE_INDEX:
            raise ValueError(
                "thumb-little round-trip phase requires feature index "
                f"{THUMB_LITTLE_DISTANCE_FEATURE_INDEX}, got width {values.shape[1]}"
            )
        distance = values[:, THUMB_LITTLE_DISTANCE_FEATURE_INDEX]
        contact_index = int(np.argmin(distance))
        if contact_index <= 0 or contact_index >= len(distance) - 1:
            phases = np.linspace(0.0, 1.0, len(values), dtype=np.float32)
        else:
            contact = float(distance[contact_index])
            inward_span = float(distance[0]) - contact
            outward_span = float(distance[-1]) - contact
            if inward_span <= max(1e-6, epsilon):
                inward = np.linspace(0.0, 1.0, contact_index + 1)
            else:
                inward = np.clip(
                    (float(distance[0]) - distance[:contact_index + 1])
                    / inward_span,
                    0.0,
                    1.0,
                )
                inward = np.maximum.accumulate(inward)
            if outward_span <= max(1e-6, epsilon):
                outward = np.linspace(0.0, 1.0, len(distance) - contact_index)
            else:
                outward = np.clip(
                    (distance[contact_index:] - contact) / outward_span,
                    0.0,
                    1.0,
                )
                outward = np.maximum.accumulate(outward)
            phases = np.empty(len(values), dtype=np.float32)
            phases[:contact_index + 1] = contact_phase * inward
            phases[contact_index:] = contact_phase + (
                1.0 - contact_phase
            ) * outward
            phases[0] = 0.0
            phases[contact_index] = contact_phase
            phases[-1] = 1.0
    else:
        steps = np.sqrt(np.mean(np.diff(values, axis=0) ** 2, axis=1))
        progress = np.concatenate((
            np.zeros(1, dtype=np.float32),
            np.cumsum(np.maximum(steps - epsilon, 0.0), dtype=np.float32),
        ))
        if float(progress[-1]) <= 1e-8:
            phases = np.linspace(0.0, 1.0, len(values), dtype=np.float32)
        else:
            phases = progress / progress[-1]

    if snap > 0.0 and len(values) > 1:
        endpoint = np.median(values[-min(window, len(values)):], axis=0)
        distances = np.sqrt(np.mean((values - endpoint) ** 2, axis=1))
        suffix_max = np.maximum.accumulate(distances[::-1])[::-1]
        stable = np.flatnonzero(suffix_max <= snap)
        if len(stable):
            phases[int(stable[0]):] = 1.0
    return np.asarray(phases, dtype=np.float32)


def primitive_template_phase_axis(
    primitive: Primitive, template: np.ndarray
) -> np.ndarray:
    """Build one phase axis from a primitive's persisted mapping settings."""
    return template_phase_axis(
        template,
        mapping=primitive.phase_mapping,
        motion_epsilon=primitive.phase_motion_epsilon,
        endpoint_snap_distance=primitive.phase_endpoint_snap_distance,
        endpoint_window=primitive.phase_endpoint_window,
        contact_fraction=primitive.phase_contact_fraction,
    )


@dataclass(frozen=True)
class MatchResult:
    primitive_id: int
    name: str
    distance: float
    second_distance: float
    confidence: float


@dataclass(frozen=True)
class PhaseMatchResult:
    """Current online action identity and normalized trajectory phase."""

    primitive_id: Optional[int]
    name: str
    phase: float
    distance: float
    second_distance: float
    locked: bool


class OnlinePhaseMatcher:
    """Causal prefix matcher for live, monotonic action-phase following.

    Each live feature advances zero to ``max_template_advance`` frames through
    every recorded template.  A class locks only after its prefix is both close
    enough and separated from the runner-up for several evaluations.  Once
    locked, phase is constrained to move forward and can advance by at most
    ``max_phase_step`` per live frame.
    """

    def __init__(
        self,
        library: ActionLibrary,
        *,
        min_lock_phase: float = 0.20,
        lock_margin: float = 0.012,
        confirm_frames: int = 3,
        max_template_advance: int = 3,
        max_phase_step: float = 0.08,
        phase_score_slack: float = 0.012,
        start_slack_ratio: float = 0.08,
        threshold_scale: float = 1.20,
    ) -> None:
        self.library = library
        self.min_lock_phase = float(np.clip(min_lock_phase, 0.0, 1.0))
        self.lock_margin = max(0.0, float(lock_margin))
        self.confirm_frames = max(1, int(confirm_frames))
        self.max_template_advance = max(1, int(max_template_advance))
        self.max_phase_step = max(1e-6, float(max_phase_step))
        self.phase_score_slack = max(0.0, float(phase_score_slack))
        self.start_slack_ratio = float(np.clip(start_slack_ratio, 0.0, 0.25))
        self.threshold_scale = max(0.1, float(threshold_scale))
        self._templates: list[tuple[Primitive, np.ndarray, np.ndarray]] = [
            (primitive, template, primitive_template_phase_axis(primitive, template))
            for primitive in library.primitives.values()
            for template in primitive.templates
        ]
        self.reset()

    def reset(self) -> None:
        self._cost_rows: list[Optional[np.ndarray]] = [None] * len(self._templates)
        self.frames = 0
        self.candidate_id: Optional[int] = None
        self.candidate_frames = 0
        self.locked_id: Optional[int] = None
        self.phase = 0.0

    def update(self, feature: np.ndarray) -> PhaseMatchResult:
        value = np.asarray(feature, dtype=np.float32).reshape(-1)
        if not len(value) or not np.all(np.isfinite(value)):
            raise ValueError("online phase feature must be finite and non-empty")
        self.frames += 1
        template_scores: list[tuple[float, float, Primitive]] = []
        local_scores: list[tuple[float, float, Primitive]] = []
        for index, (primitive, template, phase_axis) in enumerate(self._templates):
            if template.shape[1] != value.shape[0]:
                raise ValueError(
                    f"feature width {value.shape[0]} does not match template "
                    f"width {template.shape[1]}"
                )
            local = np.sqrt(np.mean((template - value[None, :]) ** 2, axis=1))
            previous = self._cost_rows[index]
            current = np.full(len(template), np.inf, dtype=np.float64)
            if previous is None:
                slack = min(
                    len(template) - 1,
                    int(np.floor(len(template) * self.start_slack_ratio)),
                )
                current[: slack + 1] = local[: slack + 1]
            else:
                for target_index in range(len(template)):
                    start = max(0, target_index - self.max_template_advance)
                    best_previous = float(np.min(previous[start : target_index + 1]))
                    if np.isfinite(best_previous):
                        current[target_index] = best_previous + float(local[target_index])
            self._cost_rows[index] = current
            normalized = current / float(self.frames)
            endpoint = int(np.argmin(normalized))
            score = float(normalized[endpoint])
            phase = float(phase_axis[endpoint])
            template_scores.append((score, phase, primitive))
            local_endpoint = int(np.argmin(local))
            local_scores.append((
                float(local[local_endpoint]),
                float(phase_axis[local_endpoint]),
                primitive,
            ))

        class_scores: list[tuple[float, float, Primitive]] = []
        allowed_id = self.locked_id
        for primitive in self.library.primitives.values():
            if allowed_id is not None and primitive.id != allowed_id:
                continue
            candidates = [item for item in template_scores if item[2].id == primitive.id]
            if candidates:
                class_scores.append(min(candidates, key=lambda item: item[0]))
        class_scores.sort(key=lambda item: item[0])
        best_distance, raw_phase, best_primitive = class_scores[0]

        if self.locked_id is None:
            all_class_scores: list[tuple[float, float, Primitive]] = []
            for primitive in self.library.primitives.values():
                candidates = [item for item in template_scores if item[2].id == primitive.id]
                all_class_scores.append(min(candidates, key=lambda item: item[0]))
            all_class_scores.sort(key=lambda item: item[0])
            best_distance, raw_phase, best_primitive = all_class_scores[0]
            second = all_class_scores[1][0] if len(all_class_scores) > 1 else np.inf
            accepted = (
                raw_phase >= self.min_lock_phase
                and best_distance <= best_primitive.threshold * self.threshold_scale
                and second - best_distance >= self.lock_margin
            )
            if accepted and self.candidate_id == best_primitive.id:
                self.candidate_frames += 1
            elif accepted:
                self.candidate_id = best_primitive.id
                self.candidate_frames = 1
            else:
                self.candidate_id = None
                self.candidate_frames = 0
            if self.candidate_frames >= self.confirm_frames:
                self.locked_id = best_primitive.id
                self.phase = float(raw_phase)
            return PhaseMatchResult(
                primitive_id=best_primitive.id if accepted else None,
                name=best_primitive.name if accepted else "searching",
                phase=float(raw_phase),
                distance=float(best_distance),
                second_distance=float(second),
                locked=self.locked_id is not None,
            )

        locked_candidates = [
            item for item in template_scores if item[2].id == self.locked_id
        ]
        best_distance = min(item[0] for item in locked_candidates)
        plausible_phases = [
            item[1]
            for item in locked_candidates
            if item[0] <= best_distance + self.phase_score_slack
        ]
        raw_phase = max(plausible_phases)
        locked_local = [item for item in local_scores if item[2].id == self.locked_id]
        best_local = min(item[0] for item in locked_local)
        local_phases = [
            item[1]
            for item in locked_local
            if item[0] <= best_local + self.phase_score_slack
        ]
        # Cumulative prefix alignment is stable; the current-frame pose term
        # helps a held-out repetition reach the correct late phase when its
        # timing differs from every training take. max_phase_step still prevents
        # a single ambiguous frame from jumping the robot through the action.
        raw_phase = max(raw_phase, max(local_phases))
        second = np.inf
        bounded = min(float(raw_phase), self.phase + self.max_phase_step)
        self.phase = max(self.phase, bounded)
        primitive = self.library.primitives[self.locked_id]
        return PhaseMatchResult(
            primitive_id=primitive.id,
            name=primitive.name,
            phase=float(self.phase),
            distance=float(best_distance),
            second_distance=float(second),
            locked=True,
        )


class LivePoseMatcher:
    """Match every live hand pose to a reversible action-library phase.

    ``OnlinePhaseMatcher`` recognizes a gesture prefix and deliberately locks a
    single, forward-only sequence.  That is useful for token recognition, but
    it cannot behave like continuous teleoperation.  This matcher instead uses
    the nearest recorded pose on every frame, applies short class hysteresis,
    and permits phase to move in either direction.
    """

    def __init__(
        self,
        library: ActionLibrary,
        *,
        selected_id: Optional[int] = None,
        match_margin: float = 0.015,
        confirm_frames: int = 2,
        switch_margin: float = 0.015,
        switch_confirm_frames: int = 2,
        reject_frames: int = 4,
        threshold_scale: float = 1.20,
        max_phase_step: float = 0.18,
        phase_smoothing: float = 0.65,
        phase_continuity_penalty: float = 0.010,
    ) -> None:
        self.library = library
        self.match_margin = max(0.0, float(match_margin))
        self.confirm_frames = max(1, int(confirm_frames))
        self.switch_margin = max(0.0, float(switch_margin))
        self.switch_confirm_frames = max(1, int(switch_confirm_frames))
        self.reject_frames = max(1, int(reject_frames))
        self.threshold_scale = max(0.1, float(threshold_scale))
        self.max_phase_step = max(1e-6, float(max_phase_step))
        self.phase_smoothing = float(np.clip(phase_smoothing, 0.0, 1.0))
        self.phase_continuity_penalty = max(0.0, float(phase_continuity_penalty))
        self._phase_axes = {
            (primitive.id, template_index): primitive_template_phase_axis(
                primitive, template
            )
            for primitive in library.primitives.values()
            for template_index, template in enumerate(primitive.templates)
        }
        self.selected_id: Optional[int] = None
        self.select(selected_id)

    def select(self, primitive_id: Optional[int]) -> None:
        """Force one class for low-latency validation, or use ``None`` for auto."""
        if primitive_id is not None and primitive_id not in self.library.primitives:
            raise ValueError(f"unknown primitive id {primitive_id}")
        self.selected_id = primitive_id
        self.reset()

    def reset(self) -> None:
        self.locked_id: Optional[int] = None
        self.phase = 0.0
        self.candidate_id: Optional[int] = None
        self.candidate_frames = 0
        self.rejected_frames = 0

    def _class_scores(
        self,
        value: np.ndarray,
        class_distance_bias: Optional[dict[int, float]] = None,
        excluded_ids: Optional[set[int]] = None,
    ) -> list[tuple[float, float, Primitive]]:
        biases = class_distance_bias or {}
        excluded = excluded_ids or set()
        scores: list[tuple[float, float, Primitive]] = []
        for primitive in self.library.primitives.values():
            if self.selected_id is not None and primitive.id != self.selected_id:
                continue
            if primitive.id in excluded:
                continue
            candidates: list[tuple[float, float]] = []
            for template_index, template in enumerate(primitive.templates):
                if template.shape[1] != value.shape[0]:
                    raise ValueError(
                        f"feature width {value.shape[0]} does not match template "
                        f"width {template.shape[1]}"
                    )
                distances = np.sqrt(
                    np.mean((template - value[None, :]) ** 2, axis=1)
                )
                phases = self._phase_axes[(primitive.id, template_index)]
                adjusted = distances
                if primitive.id == self.locked_id:
                    adjusted = distances + self.phase_continuity_penalty * np.abs(
                        phases - self.phase
                    )
                endpoint = int(np.argmin(adjusted))
                candidates.append((float(distances[endpoint]), float(phases[endpoint])))
            distance, phase = min(candidates, key=lambda item: item[0])
            bias = float(biases.get(primitive.id, 0.0))
            if not np.isfinite(bias) or bias < 0.0:
                raise ValueError("class distance biases must be finite and nonnegative")
            scores.append((distance + bias, phase, primitive))
        scores.sort(key=lambda item: item[0])
        if not scores:
            raise ValueError("class exclusions removed every live-pose candidate")
        return scores

    def _accepted(self, score: tuple[float, float, Primitive]) -> bool:
        distance, _phase, primitive = score
        return distance <= primitive.threshold * self.threshold_scale

    def _set_candidate(self, primitive_id: Optional[int]) -> None:
        if primitive_id is None:
            self.candidate_id = None
            self.candidate_frames = 0
        elif primitive_id == self.candidate_id:
            self.candidate_frames += 1
        else:
            self.candidate_id = primitive_id
            self.candidate_frames = 1

    def _update_phase(self, raw_phase: float, *, initialize: bool = False) -> None:
        raw = float(np.clip(raw_phase, 0.0, 1.0))
        if initialize:
            self.phase = raw
            return
        delta = float(np.clip(raw - self.phase, -self.max_phase_step, self.max_phase_step))
        self.phase = float(
            np.clip(self.phase + self.phase_smoothing * delta, 0.0, 1.0)
        )

    def update(
        self,
        feature: np.ndarray,
        *,
        class_distance_bias: Optional[dict[int, float]] = None,
        excluded_ids: Optional[set[int]] = None,
    ) -> PhaseMatchResult:
        """Update the match with optional class penalties and hard exclusions."""
        value = np.asarray(feature, dtype=np.float32).reshape(-1)
        if not len(value) or not np.all(np.isfinite(value)):
            raise ValueError("live pose feature must be finite and non-empty")
        excluded = excluded_ids or set()
        if self.selected_id is not None and self.selected_id in excluded:
            raise ValueError("cannot exclude the explicitly selected primitive")
        if self.locked_id in excluded:
            self.reset()
        scores = self._class_scores(value, class_distance_bias, excluded)
        best = scores[0]
        second_distance = scores[1][0] if len(scores) > 1 else np.inf

        if self.locked_id is None:
            accepted = (
                self._accepted(best)
                and second_distance - best[0] >= self.match_margin
            )
            self._set_candidate(best[2].id if accepted else None)
            required = 1 if self.selected_id is not None else self.confirm_frames
            if self.candidate_frames >= required:
                self.locked_id = best[2].id
                self.rejected_frames = 0
                self._update_phase(best[1], initialize=True)
            return PhaseMatchResult(
                primitive_id=best[2].id if accepted else None,
                name=best[2].name if accepted else "searching",
                phase=float(best[1]),
                distance=float(best[0]),
                second_distance=float(second_distance),
                locked=self.locked_id is not None,
            )

        current = next(item for item in scores if item[2].id == self.locked_id)
        current_accepted = self._accepted(current)
        switch = (
            best[2].id != self.locked_id
            and self._accepted(best)
            and (
                not current_accepted
                or current[0] - best[0] >= self.switch_margin
            )
        )
        self._set_candidate(best[2].id if switch else None)
        if switch and self.candidate_frames >= self.switch_confirm_frames:
            self.locked_id = best[2].id
            current = best
            self.rejected_frames = 0
            self._set_candidate(None)
            self._update_phase(current[1], initialize=True)
        elif not current_accepted:
            self.rejected_frames += 1
            if self.rejected_frames >= self.reject_frames:
                self.reset()
            return PhaseMatchResult(
                primitive_id=None,
                name="out_of_library",
                phase=float(self.phase),
                distance=float(current[0]),
                second_distance=float(best[0]),
                locked=False,
            )
        else:
            self.rejected_frames = 0
            self._update_phase(current[1])

        primitive = self.library.primitives[self.locked_id]
        return PhaseMatchResult(
            primitive_id=primitive.id,
            name=primitive.name,
            phase=float(self.phase),
            distance=float(current[0]),
            second_distance=float(second_distance),
            locked=True,
        )


class StreamingMatcher:
    """Recognize completed primitive templates from a live feature stream."""

    def __init__(
        self,
        library: ActionLibrary,
        *,
        evaluation_interval: int = 3,
        confirm_evaluations: int = 2,
        margin: float = 0.015,
        max_window_frames: Optional[int] = None,
    ) -> None:
        self.library = library
        longest = max(len(t) for p in library.primitives.values() for t in p.templates)
        self.buffer: deque[np.ndarray] = deque(maxlen=max_window_frames or max(12, int(longest * 1.35)))
        self.evaluation_interval = max(1, int(evaluation_interval))
        self.confirm_evaluations = max(1, int(confirm_evaluations))
        self.margin = max(0.0, float(margin))
        self.frames = 0
        self.candidate: Optional[int] = None
        self.candidate_count = 0
        self.cooldowns: dict[int, int] = {}

    def reset(self) -> None:
        self.buffer.clear()
        self.candidate = None
        self.candidate_count = 0

    def update(self, feature: np.ndarray) -> Optional[MatchResult]:
        value = np.asarray(feature, dtype=np.float32).reshape(-1)
        self.buffer.append(value)
        self.frames += 1
        self.cooldowns = {key: left - 1 for key, left in self.cooldowns.items() if left > 1}
        if self.frames % self.evaluation_interval:
            return None

        scored: list[tuple[float, Primitive]] = []
        buffered = np.asarray(self.buffer, dtype=np.float32)
        for primitive in self.library.primitives.values():
            if primitive.id in self.cooldowns:
                continue
            best = np.inf
            for template in primitive.templates:
                minimum = max(4, int(np.floor(len(template) * 0.70)))
                if len(buffered) < minimum:
                    continue
                for ratio in (0.75, 1.0, 1.25):
                    window = min(len(buffered), max(minimum, int(round(len(template) * ratio))))
                    best = min(best, dtw_distance(buffered[-window:], template))
            if np.isfinite(best):
                scored.append((float(best), primitive))
        if not scored:
            return None
        scored.sort(key=lambda pair: pair[0])
        distance, primitive = scored[0]
        second = scored[1][0] if len(scored) > 1 else np.inf
        accepted = distance <= primitive.threshold and second - distance >= self.margin
        if not accepted:
            self.candidate = None
            self.candidate_count = 0
            return None
        if self.candidate == primitive.id:
            self.candidate_count += 1
        else:
            self.candidate = primitive.id
            self.candidate_count = 1
        if self.candidate_count < self.confirm_evaluations:
            return None

        confidence = float(np.clip(1.0 - distance / max(primitive.threshold, 1e-6), 0.0, 1.0))
        result = MatchResult(primitive.id, primitive.name, distance, second, confidence)
        self.cooldowns[primitive.id] = primitive.cooldown_frames
        self.reset()
        return result


class TrajectoryExecutor:
    """Queue primitives and produce one safely step-limited command per tick."""

    def __init__(self, library: ActionLibrary, *, max_step: int = 5, blend_frames: int = 8, queue_size: int = 2):
        self.library = library
        self.max_step = max(0, int(max_step))
        self.blend_frames = max(0, int(blend_frames))
        self.queue_size = max(1, int(queue_size))
        self.queue: deque[int] = deque()
        self.active_id: Optional[int] = None
        self._trajectory: Optional[np.ndarray] = None
        self._frame = 0
        self.last_command: Optional[np.ndarray] = None

    @property
    def busy(self) -> bool:
        return self._trajectory is not None or bool(self.queue)

    def clear(self) -> None:
        self.queue.clear()
        self.active_id = None
        self._trajectory = None
        self._frame = 0

    def enqueue(self, primitive_id: int) -> bool:
        if primitive_id not in self.library.primitives:
            raise KeyError(f"unknown primitive id {primitive_id}")
        if len(self.queue) >= self.queue_size:
            return False
        if self.queue and self.queue[-1] == primitive_id:
            return False
        self.queue.append(primitive_id)
        return True

    def tick(self, observed_state: Optional[Sequence[float]] = None) -> Optional[list[int]]:
        if self._trajectory is None:
            if not self.queue:
                return None
            self._start_next(observed_state)
        assert self._trajectory is not None
        target = self._trajectory[self._frame]
        base = self.last_command
        if observed_state is not None:
            observed = np.asarray(observed_state, dtype=np.float32).reshape(-1)
            if observed.shape == (JOINT_COUNT,) and np.all(np.isfinite(observed)):
                base = observed
        if base is None or self.max_step <= 0:
            command = target.copy()
        else:
            command = base + np.clip(target - base, -self.max_step, self.max_step)
        command = np.clip(np.rint(command), 0, 255).astype(np.int32)
        command[list(RESERVED_IDX)] = 255
        self.last_command = command.astype(np.float32)
        self._frame += 1
        if self._frame >= len(self._trajectory):
            self._trajectory = None
            self.active_id = None
            self._frame = 0
        return command.tolist()

    def _start_next(self, observed_state: Optional[Sequence[float]]) -> None:
        primitive_id = self.queue.popleft()
        primitive = self.library.primitives[primitive_id]
        trajectory = primitive.trajectory
        start = self.last_command
        if observed_state is not None:
            candidate = np.asarray(observed_state, dtype=np.float32).reshape(-1)
            if candidate.shape == (JOINT_COUNT,) and np.all(np.isfinite(candidate)):
                start = candidate
        if start is None:
            start = G20_OPEN_POSE
        if self.blend_frames:
            blend = np.stack([
                (1.0 - alpha) * start + alpha * trajectory[0]
                for alpha in np.linspace(1.0 / self.blend_frames, 1.0, self.blend_frames)
            ])
            trajectory = np.concatenate((blend, trajectory[1:]), axis=0)
        self.active_id = primitive_id
        self._trajectory = trajectory
        self._frame = 0
