"""Two-finger HumanEgo action adapter for LinkerHand.

HumanEgo's released inference loop is hardware-agnostic: a policy predicts a
robot action chunk, then a rig-specific adapter turns that chunk into robot
commands.  This module is that adapter for the smallest useful LinkerHand MVP:
thumb + index only, with middle/ring/little held open.

Two action modes are supported:

``pinch3``
    ``[close, thumb_cross, index_spread]`` where close/thumb_cross are normally
    in ``[0, 1]`` and index_spread is normally in ``[-1, 1]``.  This is the
    easiest first policy head for pick/pinch tasks.

``joint7``
    ``[thumb_base, thumb_abd, thumb_opp, thumb_tip,
    index_base, index_abd, index_tip]``.  Values are normalized by default:
    flexion/opposition channels use ``[0, 1]`` and ``index_abd`` uses
    ``[-1, 1]``.  Pass ``input_range="radians"`` if the policy already outputs
    LinkerHand radians for these seven channels.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Sequence

import numpy as np

from src.finger_retarget.constants import ACTIVE_IDX, CONSTANTS, N_JOINTS, RESERVED_IDX

TWO_FINGER_IDX = (0, 5, 10, 15, 1, 6, 16)
"""Commanded semantic indices: thumb base/abd/opp/tip, index base/abd/tip."""

LOCKED_OPEN_IDX = tuple(i for i in ACTIVE_IDX if i not in TWO_FINGER_IDX)
ACTION_DIMS = {"pinch3": 3, "joint7": 7}


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _finite_float(value: object, *, name: str) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a number, got {value!r}") from exc
    if not math.isfinite(out):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return out


def _joint_limits(side: str) -> dict[int, tuple[float, float]]:
    if side not in CONSTANTS:
        raise ValueError(f"side must be 'left' or 'right', got {side!r}")
    limits: dict[int, tuple[float, float]] = {}
    for finger in CONSTANTS[side].values():
        for idx, _axis, lim in finger["base_axes"]:
            limits[int(idx)] = (float(lim[0]), float(lim[1]))
        limits[int(finger["tip_idx"])] = (
            float(finger["tip_limit"][0]),
            float(finger["tip_limit"][1]),
        )
    return limits


@dataclass(frozen=True)
class TwoFingerConfig:
    """Action scaling for the thumb/index MVP.

    The defaults are intentionally below full mechanical limits.  They give a
    conservative pinch-shaped hand in sim and leave the existing hardware range
    guards room to do their job if this is later routed to ROS.
    """

    side: str = "right"
    mode: str = "pinch3"
    input_range: str = "normalized"  # normalized | radians
    thumb_base_max: float = 0.50
    thumb_abd_max: float = 0.72
    thumb_opp_max: float = 0.95
    thumb_tip_max: float = 0.70
    index_base_max: float = 1.05
    index_spread_max: float = 0.15
    index_tip_max: float = 1.15
    pinch_thumb_base_gain: float = 0.65
    pinch_thumb_tip_gain: float = 0.85
    pinch_index_base_gain: float = 1.0
    pinch_index_tip_gain: float = 1.0

    def __post_init__(self) -> None:
        if self.side not in CONSTANTS:
            raise ValueError(f"side must be 'left' or 'right', got {self.side!r}")
        if self.mode not in ACTION_DIMS:
            raise ValueError(f"mode must be one of {sorted(ACTION_DIMS)}, got {self.mode!r}")
        if self.input_range not in ("normalized", "radians"):
            raise ValueError("input_range must be 'normalized' or 'radians'")

    @property
    def action_dim(self) -> int:
        return ACTION_DIMS[self.mode]


def _coerce_action(action: Sequence[object] | np.ndarray, *, dim: int) -> list[float]:
    vals = np.asarray(action, dtype=float).reshape(-1)
    if vals.shape[0] != dim:
        raise ValueError(f"expected action with {dim} values, got {vals.shape[0]}")
    if not np.all(np.isfinite(vals)):
        raise ValueError("action contains NaN or infinite values")
    return [float(v) for v in vals]


def _empty_l20() -> list[float]:
    return [0.0] * N_JOINTS


def _apply_pinch3(action: Sequence[object] | np.ndarray, cfg: TwoFingerConfig) -> list[float]:
    close, thumb_cross, index_spread = _coerce_action(action, dim=ACTION_DIMS["pinch3"])
    close = _clamp(close, 0.0, 1.0)
    thumb_cross = _clamp(thumb_cross, 0.0, 1.0)
    index_spread = _clamp(index_spread, -1.0, 1.0)

    q = _empty_l20()
    q[0] = cfg.thumb_base_max * cfg.pinch_thumb_base_gain * close
    q[5] = cfg.thumb_abd_max * _clamp(0.35 * close + 0.65 * thumb_cross, 0.0, 1.0)
    q[10] = cfg.thumb_opp_max * _clamp(0.45 * close + 0.55 * thumb_cross, 0.0, 1.0)
    q[15] = cfg.thumb_tip_max * cfg.pinch_thumb_tip_gain * close
    q[1] = cfg.index_base_max * cfg.pinch_index_base_gain * close
    q[6] = cfg.index_spread_max * index_spread
    q[16] = cfg.index_tip_max * cfg.pinch_index_tip_gain * close
    return q


def _scale_unit(value: float, lo: float, hi: float) -> float:
    return lo + _clamp(value, 0.0, 1.0) * (hi - lo)


def _scale_bipolar(value: float, limit: float) -> float:
    return _clamp(value, -1.0, 1.0) * abs(limit)


def _apply_joint7(action: Sequence[object] | np.ndarray, cfg: TwoFingerConfig) -> list[float]:
    vals = _coerce_action(action, dim=ACTION_DIMS["joint7"])
    q = _empty_l20()
    if cfg.input_range == "radians":
        for idx, value in zip(TWO_FINGER_IDX, vals):
            q[idx] = value
        return q

    q[0] = _scale_unit(vals[0], 0.0, cfg.thumb_base_max)
    q[5] = _scale_unit(vals[1], 0.0, cfg.thumb_abd_max)
    q[10] = _scale_unit(vals[2], 0.0, cfg.thumb_opp_max)
    q[15] = _scale_unit(vals[3], 0.0, cfg.thumb_tip_max)
    q[1] = _scale_unit(vals[4], 0.0, cfg.index_base_max)
    q[6] = _scale_bipolar(vals[5], cfg.index_spread_max)
    q[16] = _scale_unit(vals[6], 0.0, cfg.index_tip_max)
    return q


def clip_l20_command(joint_rad: Sequence[object] | np.ndarray, side: str) -> list[float]:
    """Return a valid 20-vector with only thumb/index movable.

    Any middle/ring/little command is discarded.  Reserved channels are forced to
    zero.  Remaining active channels are clipped to the generated URDF limits.
    """

    q = np.asarray(joint_rad, dtype=float).reshape(-1)
    if q.shape[0] != N_JOINTS:
        raise ValueError(f"joint_rad must have {N_JOINTS} entries, got {q.shape[0]}")
    if not np.all(np.isfinite(q)):
        raise ValueError("joint_rad contains NaN or infinite values")

    limits = _joint_limits(side)
    out = [0.0] * N_JOINTS
    for idx in TWO_FINGER_IDX:
        lo, hi = limits[idx]
        out[idx] = _clamp(float(q[idx]), lo, hi)
    for idx in LOCKED_OPEN_IDX + tuple(RESERVED_IDX):
        out[idx] = 0.0
    return out


def action_to_l20(action: Sequence[object] | np.ndarray,
                  cfg: TwoFingerConfig | None = None) -> dict:
    """Convert one policy action to the canonical ``l20_targets``-like dict."""

    cfg = cfg or TwoFingerConfig()
    if cfg.mode == "pinch3":
        q = _apply_pinch3(action, cfg)
    elif cfg.mode == "joint7":
        q = _apply_joint7(action, cfg)
    else:  # pragma: no cover - guarded by config validation
        raise ValueError(f"unsupported action mode {cfg.mode!r}")

    return {
        "side": cfg.side,
        "joint_rad": clip_l20_command(q, cfg.side),
        "active_idx": list(ACTIVE_IDX),
        "clamped": True,
        "mode": cfg.mode,
        "action_dim": cfg.action_dim,
        "two_finger_idx": list(TWO_FINGER_IDX),
    }


def lock_candidate_to_two_finger(candidate: Mapping[str, object], *,
                                 landmarks=None) -> dict:
    """Freeze a full-hand retarget candidate down to thumb + index.

    This is the live-camera bridge for step 1 of the HumanEgo/LinkerHand plan:
    keep the existing MediaPipe -> retarget solver, but discard every channel
    except the two-finger subset.  The return value keeps the normal
    ``l20_targets`` shape so ``src.viz.core.teleop_command`` can still run the
    safety filter after the lock.
    """

    if "joint_rad" not in candidate:
        raise ValueError("candidate must contain 'joint_rad'")
    side = str(candidate.get("side", "right"))
    out = dict(candidate)
    out["side"] = side
    out["joint_rad"] = clip_l20_command(candidate["joint_rad"], side)
    out["active_idx"] = list(ACTIVE_IDX)
    out["clamped"] = True
    out["two_finger_idx"] = list(TWO_FINGER_IDX)
    return out


def _extract_action(record: object, *, line_no: int) -> tuple[list[float], float | None]:
    if isinstance(record, Mapping):
        raw_action = record.get("action", record.get("actions", record.get("joint7")))
        if raw_action is None:
            raise ValueError(f"line {line_no}: expected an 'action' field")
        t_raw = record.get("t", record.get("time"))
    else:
        raw_action = record
        t_raw = None
    if not isinstance(raw_action, Sequence) or isinstance(raw_action, (str, bytes)):
        raise ValueError(f"line {line_no}: action must be a numeric sequence")
    action = [_finite_float(v, name=f"line {line_no} action") for v in raw_action]
    t = None if t_raw is None else _finite_float(t_raw, name=f"line {line_no} t")
    return action, t


def iter_action_records(path: str | Path) -> Iterator[tuple[list[float], float | None]]:
    """Yield ``(action, t)`` from JSON, JSONL, or NumPy trajectory files."""

    p = Path(path)
    suffix = p.suffix.lower()
    if suffix == ".npy":
        arr = np.load(p)
        for i, row in enumerate(np.asarray(arr)):
            action = [_finite_float(v, name=f"row {i} action") for v in np.asarray(row).reshape(-1)]
            yield action, None
        return
    if suffix == ".npz":
        data = np.load(p)
        key = "actions" if "actions" in data else sorted(data.files)[0]
        for i, row in enumerate(np.asarray(data[key])):
            action = [_finite_float(v, name=f"row {i} action") for v in np.asarray(row).reshape(-1)]
            yield action, None
        return
    if suffix == ".jsonl":
        with p.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                yield _extract_action(json.loads(line), line_no=line_no)
        return

    with p.open("r", encoding="utf-8") as f:
        doc = json.load(f)
    records = doc.get("actions", doc) if isinstance(doc, Mapping) else doc
    if not isinstance(records, Sequence):
        raise ValueError("JSON trajectory must be a list or contain an 'actions' list")
    for i, record in enumerate(records):
        yield _extract_action(record, line_no=i + 1)


def demo_actions(mode: str = "pinch3", *, frames: int = 90) -> Iterable[list[float]]:
    """Generate a smooth open-close-open trajectory for smoke tests."""

    if mode not in ACTION_DIMS:
        raise ValueError(f"mode must be one of {sorted(ACTION_DIMS)}, got {mode!r}")
    n = max(2, int(frames))
    for i in range(n):
        phase = i / (n - 1)
        close = 0.5 - 0.5 * math.cos(2.0 * math.pi * phase)
        if mode == "pinch3":
            yield [close, min(1.0, close * 1.15), 0.0]
        else:
            yield [close * 0.65, close * 0.75, close, close * 0.85,
                   close, 0.0, close]
