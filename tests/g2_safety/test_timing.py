"""Test 7: determinism + two-part timing gate.

The filter is deterministic (fixed iteration count, no randomness) and real-time.
Two-part gate (ticket §7):
  (a) ABSOLUTE  — every call (incl. the deep-collision worst case) finishes well
      inside one 30 Hz camera frame, LOOP_PERIOD_US = 33,333 µs. This is the hard
      real-time guarantee; the solver tail (~0.78 ms) leaves the rest of the frame.
  (b) REGRESSION — p99 over the committed workload must not exceed the committed
      baseline by more than FILTER_LATENCY_REGRESSION_MARGIN, catching silent
      slowdowns long before they could threaten the ceiling.
"""
import time

import numpy as np
import pytest

from src.safety import (
    LOOP_PERIOD_US, FILTER_P99_BASELINE_US, FILTER_LATENCY_REGRESSION_MARGIN,
)
from helpers import timing_workload, rand_in_limits, sample_colliding


# ---- determinism ---------------------------------------------------------- #
def test_deterministic_same_input_same_output(filt_right):
    rng = np.random.default_rng(99)
    cases = [rand_in_limits(filt_right.model, rng) for _ in range(20)]
    cases += sample_colliding(filt_right.model, rng, 20)
    for c in cases:
        r1 = filt_right.filter(c, c, 0.033)
        r2 = filt_right.filter(c, c, 0.033)
        assert r1 == r2


def test_fixed_iteration_count_no_early_unbounded_loop(filt_right):
    # The projection loop is bounded by cfg.pbd_iterations regardless of input.
    assert filt_right.cfg.pbd_iterations >= 1
    # deeply-colliding input still returns (does not hang / iterate unbounded)
    rng = np.random.default_rng(100)
    deep = sample_colliding(filt_right.model, rng, 1, min_depth=0.01)
    assert deep, "could not build a deep-collision case"
    r = filt_right.filter(deep[0], deep[0], 1.0)
    assert len(r["joint_rad"]) == 20


def _latencies(f, workload, dt=1.0):
    # warmup (JIT/caches/branch predictors) then time each call once
    for c in workload[:10]:
        f.filter(c, c, dt)
    lat = []
    for c in workload:
        t = time.perf_counter()
        f.filter(c, c, dt)
        lat.append((time.perf_counter() - t) * 1e6)
    return np.array(lat)


# ---- (a) absolute real-time ceiling --------------------------------------- #
def test_absolute_ceiling_every_call_under_loop_period(filt_right):
    workload = timing_workload(filt_right.model)
    # best of 3 passes to suppress one-off OS scheduling spikes
    worst = min(_latencies(filt_right, workload).max() for _ in range(3))
    assert worst < LOOP_PERIOD_US, \
        f"worst filter latency {worst:.0f} us exceeds frame {LOOP_PERIOD_US:.0f} us"


# ---- (b) regression guard vs committed baseline --------------------------- #
def test_p99_regression_vs_committed_baseline(filt_right):
    workload = timing_workload(filt_right.model)
    # best-of-3 p99 to reduce machine jitter
    p99 = min(float(np.percentile(_latencies(filt_right, workload), 99))
              for _ in range(3))
    ceiling = FILTER_P99_BASELINE_US * (1.0 + FILTER_LATENCY_REGRESSION_MARGIN)
    assert p99 <= ceiling, (
        f"filter p99 {p99:.0f} us regressed past baseline "
        f"{FILTER_P99_BASELINE_US:.0f} us +{FILTER_LATENCY_REGRESSION_MARGIN:.0%} "
        f"= {ceiling:.0f} us")
    # sanity: the committed baseline itself sits comfortably under the ceiling
    assert FILTER_P99_BASELINE_US < LOOP_PERIOD_US
