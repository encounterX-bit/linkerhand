"""The single per-frame teleop seam shared by the live and replay loops.

``hand_landmarks (hand_base)`` -> ``retarget()`` -> ``safety.filter()`` -> the
16-DoF command. Both ``run_live`` and ``run_camera_free`` drive the exact same
``teleop_command`` + ``drive`` here, so the camera-free replay exercises the
*identical* code path as the live mirror (that equivalence is what the headless
test pins). ``retarget`` and ``safety.filter`` are imported READ-ONLY — this
module is pure orchestration over them, no retarget/filter logic is reimplemented.
"""
from __future__ import annotations

from typing import Callable, Iterable, List, Optional, Tuple

import numpy as np

from src.finger_retarget import retarget          # read-only system under test
from src import safety                              # read-only G2 guard

# Default control/camera period (30 Hz), matching the G2 closed-loop control_hz.
DEFAULT_DT = 1.0 / 30.0


def teleop_command(landmarks, side: str, prev_safe=None, dt: float = DEFAULT_DT,
                   *, use_filter: bool = True,
                   candidate_adjust: Optional[Callable[..., dict]] = None) -> dict:
    """One frame: landmarks -> candidate (retarget) -> safe command (filter).

    Returns ``{command: list[20], candidate: dict, safe: dict|None,
    modified: bool, reason: str|None}``. ``command`` is the 20-vector actually
    applied to the joints (post-filter when ``use_filter``; the raw retarget
    output otherwise). ``prev_safe`` is the previous frame's ``safe`` dict (or
    None), threaded for the filter's rate-limit guard — exactly as the G2
    closed-loop does.
    """
    cand = retarget(landmarks, side=side)
    if candidate_adjust is not None:
        cand = candidate_adjust(cand, landmarks=landmarks)
    if not use_filter:
        return {"command": list(cand["joint_rad"]), "candidate": cand,
                "safe": None, "modified": False, "reason": None}
    safe = safety.filter(cand, prev_safe, dt, side=side)
    return {"command": list(safe["joint_rad"]), "candidate": cand,
            "safe": safe, "modified": bool(safe["modified"]),
            "reason": safe["reason"]}


def drive(model, stream: Iterable[Tuple[str, object, float]], *,
          use_filter: bool = True, dt: float = DEFAULT_DT,
          on_record: Optional[Callable[[dict, dict], None]] = None,
          candidate_adjust: Optional[Callable[..., dict]] = None,
          debug_callback: Optional[Callable[[int, str, object, dict], None]] = None) -> List[dict]:
    """Run the teleop loop over a ``(side, landmarks, t)`` stream.

    For each frame: compute the command via :func:`teleop_command`, apply it to
    ``model`` (``model.set_joints``; pass ``None`` for a headless no-render run),
    and accumulate a record. ``prev_safe`` is threaded across frames and reset
    whenever the hand side changes (the per-side filter rejects a prev_safe from
    the other side — same reset rule as perception's smoother).

    Returns the list of per-frame records:
    ``{frame, t, side, command, modified, reason}``.
    """
    prev_safe = None
    last_side: Optional[str] = None
    records: List[dict] = []
    for frame_i, (side, landmarks, t) in enumerate(stream):
        if side != last_side:
            prev_safe = None
            last_side = side
        out = teleop_command(landmarks, side, prev_safe, dt, use_filter=use_filter,
                             candidate_adjust=candidate_adjust)
        if out["safe"] is not None:
            prev_safe = out["safe"]
        if model is not None:
            model.set_joints(out["command"])
        rec = {"frame": frame_i, "t": float(t), "side": side,
               "command": out["command"], "modified": out["modified"],
               "reason": out["reason"]}
        records.append(rec)
        if on_record is not None:
            on_record(rec, out)
        if debug_callback is not None:
            debug_callback(frame_i, side, landmarks, out)
    return records
