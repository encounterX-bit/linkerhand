"""Compact MediaPipe-vs-sim shape diagnostics for live retarget tuning."""
from __future__ import annotations

from typing import Dict

import numpy as np

from src.kinematics import L20FK
from src.kinematics.conventions import FINGER_ORDER, FINGERS


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-9:
        return np.zeros_like(v, dtype=float)
    return np.asarray(v, dtype=float) / n


def _angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    ua, ub = _unit(a), _unit(b)
    if not np.any(ua) or not np.any(ub):
        return 0.0
    cos = float(np.clip(ua @ ub, -1.0, 1.0))
    return float(np.degrees(np.arccos(cos)))


def _curl_deg(prox: np.ndarray, dist: np.ndarray) -> float:
    return _angle_deg(prox, dist)


class MatchDebugger:
    """Print normalized MediaPipe/sim per-finger geometry every N frames."""

    def __init__(self, *, period: int = 15):
        self.period = max(1, int(period))
        self._fk: Dict[str, L20FK] = {}

    def __call__(self, frame_i: int, side: str, landmarks, out: dict) -> None:
        if frame_i % self.period != 0:
            return
        lm = np.asarray(landmarks, dtype=float)
        if lm.shape != (21, 3) or not np.all(np.isfinite(lm)):
            return

        fk = self._fk.get(side)
        if fk is None:
            fk = L20FK(side)
            self._fk[side] = fk
        q = np.asarray(out["command"], dtype=float)
        fk.set_joint_rad(q)

        parts = []
        for name in FINGER_ORDER:
            spec = FINGERS[name]
            a, b, _c, d = spec.landmarks
            mp_prox = lm[b] - lm[a]
            mp_dist = lm[d] - lm[b]
            sim_prox, sim_dist = fk.segment_dirs(spec)

            prox_err = _angle_deg(mp_prox, sim_prox)
            dist_err = _angle_deg(mp_dist, sim_dist)
            mp_curl = _curl_deg(mp_prox, mp_dist)
            sim_curl = _curl_deg(sim_prox, sim_dist)
            curl_err = sim_curl - mp_curl
            joints = ",".join(f"q{i}={q[i]:.2f}" for i in spec.active_indices)
            parts.append(
                f"{name[:1]}:p{prox_err:.0f} d{dist_err:.0f} "
                f"c{curl_err:+.0f}({mp_curl:.0f}->{sim_curl:.0f}) {joints}"
            )

        print(f"[match] frame={frame_i} side={side} " + " | ".join(parts),
              flush=True)


def make_match_debugger(enabled: bool, *, period: int = 15):
    return MatchDebugger(period=period) if enabled else None
