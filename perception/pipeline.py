"""The perception pipeline: ``RawDetection`` stream -> ``hand_landmarks``.

Per frame:
  1. resolve handedness -> L20 side (or use a forced side),
  2. transform into the ADR-0003 hand_base frame,
  3. one-euro smooth (on by default; input-side only, never in the solver),
  4. robustness: on no-detection / low-confidence / non-finite, hold the last
     good output and flag it -- never emit garbage or NaN,
  5. carry a depth-confidence signal so weak-z frames are visible downstream.

Output is a ``ProcessedFrame``; ``.to_contract()`` yields a dict valid against
contracts/hand_landmarks.schema.json.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Iterator, List, Optional

import numpy as np

from .frame import to_hand_base
from .handedness import to_l20_side
from .one_euro import LandmarkSmoother, OneEuroConfig
from .source import HandSource, RawDetection


@dataclass
class ProcessedFrame:
    side: str
    landmarks: np.ndarray            # (21, 3) hand_base frame, finite
    t: float
    detected: bool                   # True if backed by a fresh detection
    score: float
    depth_confidence: float
    held: bool = False               # True if reused from the last good frame
    warnings: List[str] = field(default_factory=list)

    def to_contract(self) -> dict:
        """A dict valid against the hand_landmarks contract (+ diagnostic keys)."""
        lm = np.asarray(self.landmarks, dtype=float)
        if lm.shape != (21, 3) or not np.all(np.isfinite(lm)):
            raise ValueError("refusing to emit non-(21,3)/non-finite landmarks")
        return {
            "side": self.side,
            "frame": "hand_base",
            "t": float(self.t),
            "landmarks": [[float(v) for v in p] for p in lm],
            # diagnostics (extra keys; contract allows them):
            "detected": bool(self.detected),
            "held": bool(self.held),
            "score": float(self.score),
            "depth_confidence": float(self.depth_confidence),
            "warnings": list(self.warnings),
        }


class HandPipeline:
    def __init__(
        self,
        source: HandSource,
        *,
        smoothing: bool = True,
        one_euro: Optional[OneEuroConfig] = None,
        image_mirrored: bool = False,
        force_side: Optional[str] = None,
        min_score: float = 0.5,
        depth_warn: float = 0.5,
    ):
        self.source = source
        self.image_mirrored = image_mirrored
        self.force_side = force_side
        self.min_score = min_score
        self.depth_warn = depth_warn
        cfg = replace(one_euro) if one_euro is not None else OneEuroConfig()
        cfg.enabled = cfg.enabled and smoothing
        self._smoother = LandmarkSmoother(cfg)
        self._last: Optional[ProcessedFrame] = None

    # -- per-frame -------------------------------------------------------- #
    def process(self, det: Optional[RawDetection]) -> Optional[ProcessedFrame]:
        """Process one detection. Returns a ProcessedFrame, or None if there is
        no fresh detection AND no prior good frame to hold."""
        fresh = (
            det is not None
            and det.ok
            and det.landmarks is not None
            and det.score >= self.min_score
        )

        if fresh:
            try:
                side = self.force_side or to_l20_side(det.handedness, self.image_mirrored)
                hb = to_hand_base(det.landmarks, side)
            except ValueError as e:
                return self._hold(det, f"bad_frame:{e}")
            if not np.all(np.isfinite(hb)):
                return self._hold(det, "non_finite_transform")

            # reset the smoother if the hand side changed
            if self._last is not None and self._last.side != side:
                self._smoother.reset()
            hb = self._smoother.filter(hb, det.t)
            if not np.all(np.isfinite(hb)):
                return self._hold(det, "non_finite_smoothed")

            warnings: List[str] = []
            if det.depth_confidence < self.depth_warn:
                warnings.append("low_depth_confidence")
            pf = ProcessedFrame(
                side=side,
                landmarks=hb,
                t=det.t,
                detected=True,
                score=det.score,
                depth_confidence=det.depth_confidence,
                held=False,
                warnings=warnings,
            )
            self._last = pf
            return pf

        # not fresh: hold last good if we have one
        reason = "no_detection" if (det is None or not det.ok) else "low_confidence"
        return self._hold(det, reason)

    def _hold(self, det: Optional[RawDetection], reason: str) -> Optional[ProcessedFrame]:
        if self._last is None:
            return None  # nothing good yet -> emit nothing (never garbage)
        t = det.t if det is not None else self._last.t
        warnings = ["held", reason]
        if self._last.depth_confidence < self.depth_warn:
            warnings.append("low_depth_confidence")
        return ProcessedFrame(
            side=self._last.side,
            landmarks=self._last.landmarks.copy(),
            t=t,
            detected=False,
            score=det.score if det is not None else 0.0,
            depth_confidence=self._last.depth_confidence,
            held=True,
            warnings=warnings,
        )

    # -- whole stream ----------------------------------------------------- #
    def run(self) -> Iterator[ProcessedFrame]:
        for det in self.source:
            pf = self.process(det)
            if pf is not None:
                yield pf
