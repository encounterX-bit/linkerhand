"""The inline G2 safety guard: ``candidate l20_targets -> SAFE l20_targets``.

A *projection*, not a checker. Given a candidate config, the previous SAFE
config, and the frame ``dt``, it returns the nearest config that is

  1. within ``hardware/LIMITS.md`` joint ranges (drivers AND mimic dependents),
  2. reserved idx 11-14 == 0,
  3. within the per-joint rate limit relative to ``prev_safe`` (so a perception
     glitch / teleport cannot command a jump),
  4. self-collision-free under the capsule + palm model (XPBD-style fixed-
     iteration non-penetration projection over adjacent fingers, thumb-vs-finger
     and fingertip-vs-palm).

Guards 1-3 define a per-joint FEASIBLE BOX; the collision projection only ever
moves inside that box, so the output always satisfies every guard regardless of
how the projection converges. Fixed iteration count -> deterministic + real-time.

Pure, deterministic, sim-free, no hardware path. FK + conventions come from the
single ``src/kinematics`` authority; the force-clamp and watchdog SPECS live in
``config`` and are never actuated here.
"""
from __future__ import annotations

import numpy as np

from src.kinematics import RESERVED_IDX, ACTIVE_IDX

from .config import SafetyConfig, DEFAULT_CONFIG
from .collision_model import CollisionModel

_RESERVED = tuple(RESERVED_IDX)
_ACTIVE = tuple(ACTIVE_IDX)


def _coerce(target, default_side):
    """Return (joint_rad ndarray[20], side) from a list or an l20_targets dict."""
    side = default_side
    if isinstance(target, dict):
        side = target.get("side", side)
        arr = target["joint_rad"]
    else:
        arr = target
    a = np.asarray(arr, dtype=float).reshape(-1)
    if a.shape[0] != 20:
        raise ValueError(f"joint_rad must have 20 entries, got {a.shape[0]}")
    return a.copy(), side


class SafetyFilter:
    """A reusable filter for one side. Holds the collision model + effective
    (mimic-tightened) joint limits, both derived from src/kinematics."""

    def __init__(self, side: str = "right", cfg: SafetyConfig = DEFAULT_CONFIG):
        self.side = side
        self.cfg = cfg
        self.model = CollisionModel(side)
        self.model.configure(adjacent=cfg.check_adjacent_fingers,
                              thumb=cfg.check_thumb_vs_fingers,
                              palm=cfg.check_fingertip_vs_palm)
        self._lo, self._hi = self._effective_limits()

    def _effective_limits(self):
        """Per-active-idx (lo, hi), tightened so every mimic DEPENDENT also stays
        in range (mimic consistency on the driver joints; ratios from FK)."""
        fk = self.model.fk
        active = fk.active_limits()              # {idx: (lo, hi)}
        lo = np.full(20, 0.0)
        hi = np.full(20, 0.0)
        for idx, (l, h) in active.items():
            lo[idx], hi[idx] = l, h
        name_to_idx = self.model._name_to_idx
        for mj, (driver, mult, off) in fk.mimics.items():
            if driver not in name_to_idx or mj not in fk.limits:
                continue
            didx = name_to_idx[driver]
            dlo, dhi = fk.limits[mj]
            if mult > 0:
                lo[didx] = max(lo[didx], (dlo - off) / mult)
                hi[didx] = min(hi[didx], (dhi - off) / mult)
            elif mult < 0:
                lo[didx] = max(lo[didx], (dhi - off) / mult)
                hi[didx] = min(hi[didx], (dlo - off) / mult)
        return lo, hi

    # -- guards -------------------------------------------------------------- #
    def _static_clamp(self, q):
        """Sanitise non-finite (NaN/inf glitch) -> lower limit; reserved -> 0;
        clamp active to effective limits. Returns (q, changed)."""
        out = q.copy()
        # NaN/inf can never reach a command: fall back to the (safe, open-ish)
        # lower limit. np.clip alone does NOT remove NaN.
        bad = ~np.isfinite(out)
        if bad.any():
            out[bad] = self._lo[bad]
        out[list(_RESERVED)] = 0.0
        np.clip(out, self._lo, self._hi, out=out)
        changed = bad.any() or not np.array_equal(out, q)
        return out, bool(changed)

    def _feasible_box(self, prev_safe, dt):
        """Per-joint [box_lo, box_hi] = limits intersected with the rate band
        around prev_safe. Reserved collapse to [0, 0]."""
        box_lo = self._lo.copy()
        box_hi = self._hi.copy()
        if prev_safe is not None and dt is not None and dt > 0.0:
            step = self.cfg.max_joint_vel_rad_s * dt
            band_lo = prev_safe - step
            band_hi = prev_safe + step
            box_lo = np.maximum(box_lo, band_lo)
            box_hi = np.maximum(box_lo, np.minimum(box_hi, band_hi))
        box_lo[list(_RESERVED)] = 0.0
        box_hi[list(_RESERVED)] = 0.0
        return box_lo, box_hi

    def _project_collision(self, q, box_lo, box_hi):
        """XPBD-style fixed-iteration non-penetration projection inside the box.
        Returns (q, iterations_that_found_a_violation)."""
        margin = self.cfg.separation_margin_m
        scale = self.cfg.pbd_step_scale
        active_hits = 0
        for _ in range(self.cfg.pbd_iterations):
            pens = self.model.penetrations(q, margin)
            if not pens:
                break
            active_hits += 1
            dq = np.zeros(20)
            for p in pens:
                g = p.grad
                gg = float(g @ g)
                if gg < 1e-12:
                    continue
                dq += (scale * p.depth / gg) * g
            q = np.clip(q + dq, box_lo, box_hi)
            q[list(_RESERVED)] = 0.0
        return q, active_hits

    # -- main ---------------------------------------------------------------- #
    def filter(self, candidate, prev_safe, dt: float) -> dict:
        cand, side = _coerce(candidate, self.side)
        if side != self.side:
            raise ValueError(f"filter built for {self.side!r}, got side={side!r}")
        prev = None
        if prev_safe is not None:
            prev, _ = _coerce(prev_safe, self.side)

        reasons = []

        # 1+2. static clamp (limits, reserved, sanitise non-finite).
        q0, clamp_changed = self._static_clamp(cand)
        if clamp_changed:
            reasons.append("limits")

        # 3. STABLE collision target: project the candidate inside the FULL-limits
        #    box (rate-independent -> a pure function of the candidate). Projecting
        #    here, before rate limiting, is what kills boundary chatter: the target
        #    a held candidate converges to does not depend on prev_safe.
        lim_lo, lim_hi = self._feasible_box(None, None)
        q_target, hits1 = self._project_collision(q0, lim_lo, lim_hi)

        # 4. rate limit: move toward the stable target within the band around
        #    prev_safe (a perception glitch / teleport cannot jump).
        box_lo, box_hi = self._feasible_box(prev, dt)
        q_rate = np.clip(q_target, box_lo, box_hi)
        clipped = (prev is not None and dt and dt > 0.0
                   and not np.allclose(q_rate, q_target, atol=0.0))

        # 5. If the rate limit pulled us OFF the stable target, the clipped config
        #    may re-enter collision -> re-project inside the (small) band so the
        #    OUTPUT is always collision-free mid-approach. When NOT clipped,
        #    q_rate == q_target is already projected: use it directly (single-pass
        #    cost, and the target is rate-independent so a held candidate cannot
        #    chatter).
        hits2 = 0
        if clipped:
            reasons.append("rate_limit")
            q, hits2 = self._project_collision(q_rate, box_lo, box_hi)
        else:
            q = q_rate
        if hits1 > 0 or hits2 > 0:
            reasons.append("self_collision")

        q[list(_RESERVED)] = 0.0
        modified = not np.allclose(q, cand, atol=self.cfg.eps_rad, rtol=0.0)
        reason = ",".join(reasons) if (modified and reasons) else None
        return {
            "joint_rad": [float(x) for x in q],
            "clamped": True,
            "modified": bool(modified),
            "reason": reason,
        }


# --- module-level convenience: a cached default filter per side ------------- #
_DEFAULT_FILTERS: dict = {}


def get_filter(side: str = "right", cfg: SafetyConfig = DEFAULT_CONFIG) -> SafetyFilter:
    key = (side, id(cfg))
    f = _DEFAULT_FILTERS.get(key)
    if f is None:
        f = SafetyFilter(side, cfg)
        _DEFAULT_FILTERS[key] = f
    return f


def filter(candidate, prev_safe, dt: float, side: str = "right",
           cfg: SafetyConfig = DEFAULT_CONFIG) -> dict:
    """Locked seam (G2 ticket): ``filter(candidate, prev_safe, dt) -> dict``.

    ``side`` is inferred from the candidate if it is an l20_targets dict,
    otherwise taken from this argument (default 'right'). Returns
    ``{joint_rad[20], clamped: True, modified: bool, reason: str|None}``.
    """
    s = candidate.get("side", side) if isinstance(candidate, dict) else side
    return get_filter(s, cfg).filter(candidate, prev_safe, dt)
