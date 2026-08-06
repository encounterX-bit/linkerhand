"""src/safety — the inline G2 guard between the retargeter and the command.

Public API:
    filter(candidate, prev_safe, dt[, side, cfg]) -> dict   # the locked seam
    SafetyFilter(side, cfg)                                  # reusable instance
    SafetyConfig / ForceClampSpec / WatchdogSpec / DEFAULT_CONFIG
    CollisionModel                                          # capsule + palm model

NEVER actuates hardware. Force-clamp and watchdog are SPECS only (config), for
``comms`` to enforce at G3. FK + conventions come from ``src/kinematics``.

This module requires HUMAN REVIEW before merge (root CLAUDE.md: changes to
src/safety need explicit human review).
"""
from .config import (
    SafetyConfig, ForceClampSpec, WatchdogSpec, DEFAULT_CONFIG,
    LOOP_PERIOD_US, FILTER_P99_BASELINE_US, FILTER_LATENCY_REGRESSION_MARGIN,
)
from .collision_model import CollisionModel
from .filter import filter, get_filter, SafetyFilter

__all__ = [
    "filter", "get_filter", "SafetyFilter",
    "SafetyConfig", "ForceClampSpec", "WatchdogSpec", "DEFAULT_CONFIG",
    "CollisionModel",
    "LOOP_PERIOD_US", "FILTER_P99_BASELINE_US", "FILTER_LATENCY_REGRESSION_MARGIN",
]
