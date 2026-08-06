"""One-euro filter for landmark smoothing (Casiez et al., CHI 2012).

Monocular depth (z) is jittery, so the perception output is smoothed BEFORE it
reaches the solver -- ``finger_retarget`` stays a pure, unsmoothed function and
the solver hot path carries no filtering (see the perception ticket).

The one-euro filter is a first-order low-pass whose cutoff adapts to speed: it
removes jitter when the hand is still (low ``min_cutoff``) yet adds little lag
when the hand moves fast (``beta`` raises the cutoff with velocity).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _alpha(cutoff: float, dt: float) -> float:
    tau = 1.0 / (2.0 * np.pi * cutoff)
    return 1.0 / (1.0 + tau / dt)


@dataclass
class OneEuroConfig:
    """Filter gains. Defaults tuned for ~30 Hz metre-scale landmarks."""

    min_cutoff: float = 1.5   # Hz; lower -> smoother but laggier when still
    beta: float = 0.05        # speed coefficient; higher -> less lag when moving
    d_cutoff: float = 1.0     # Hz; cutoff of the derivative low-pass
    enabled: bool = True

    def __post_init__(self) -> None:
        if self.min_cutoff <= 0.0:
            raise ValueError("min_cutoff must be > 0")
        if self.beta < 0.0:
            raise ValueError("beta must be >= 0")
        if self.d_cutoff <= 0.0:
            raise ValueError("d_cutoff must be > 0")


class OneEuroVectorFilter:
    """One-euro filter over a flat vector of independent scalar channels."""

    def __init__(self, dim: int, config: OneEuroConfig | None = None):
        self.cfg = config or OneEuroConfig()
        self.dim = dim
        self._x_prev: np.ndarray | None = None
        self._dx_prev = np.zeros(dim)
        self._t_prev: float | None = None

    def reset(self) -> None:
        self._x_prev = None
        self._dx_prev = np.zeros(self.dim)
        self._t_prev = None

    def __call__(self, x, t: float) -> np.ndarray:
        return self.filter(x, t)

    def filter(self, x, t: float) -> np.ndarray:
        x = np.asarray(x, dtype=float).reshape(-1)
        if x.shape[0] != self.dim:
            raise ValueError(f"expected dim {self.dim}, got {x.shape[0]}")
        if not self.cfg.enabled:
            return x.copy()

        if self._x_prev is None or self._t_prev is None:
            self._x_prev = x.copy()
            self._t_prev = t
            self._dx_prev = np.zeros(self.dim)
            return x.copy()

        dt = t - self._t_prev
        if dt <= 0.0:
            # Out-of-order / duplicate timestamp: return last estimate, no update.
            return self._x_prev.copy()

        dx = (x - self._x_prev) / dt
        a_d = _alpha(self.cfg.d_cutoff, dt)
        edx = a_d * dx + (1.0 - a_d) * self._dx_prev
        cutoff = self.cfg.min_cutoff + self.cfg.beta * np.abs(edx)
        a = _alpha(cutoff, dt)
        x_hat = a * x + (1.0 - a) * self._x_prev

        self._x_prev = x_hat
        self._dx_prev = edx
        self._t_prev = t
        return x_hat.copy()


class LandmarkSmoother:
    """One-euro smoothing over a (21, 3) landmark array."""

    def __init__(self, config: OneEuroConfig | None = None):
        self._filt = OneEuroVectorFilter(21 * 3, config)

    @property
    def enabled(self) -> bool:
        return self._filt.cfg.enabled

    def reset(self) -> None:
        self._filt.reset()

    def filter(self, landmarks, t: float) -> np.ndarray:
        lm = np.asarray(landmarks, dtype=float)
        out = self._filt.filter(lm.reshape(-1), t)
        return out.reshape(21, 3)
