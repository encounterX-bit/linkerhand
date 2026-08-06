"""G2 Test 1 — loop rate (two-part gate).

The whole landmarks -> retarget() -> safety.filter() -> command -> step loop must run
at camera rate. ``compute`` per tick = retarget + filter + sim-step (the real-time
work); the latency delay buffer is MODELED delay and does NOT count.

  (a) ABSOLUTE: p99 compute < LOOP_PERIOD (33,333 µs = one 30 Hz frame) — the hard
      real-time guarantee.
  (b) REGRESSION: p99 compute < committed baseline * (1 + margin) — catches a silent
      slowdown that is still under the ceiling.
"""
import os

import numpy as np

from src.sim import ClosedLoopSim


def _p99_compute(frames, side="right"):
    cl = ClosedLoopSim(side, latency_s=0.0, use_filter=True)
    try:
        recs = cl.run(frames)
    finally:
        cl.close()
    return np.percentile([r.compute_us for r in recs], 99), recs


def test_loop_rate_absolute_and_regression(frames_right, baseline):
    lr = baseline["loop_rate"]
    period = lr["period_us"]
    base = lr["p99_compute_baseline_us"]
    margin = float(os.environ.get("LOOP_REGRESSION_MARGIN", lr["regression_margin"]))

    # best-of-3 to suppress scheduler jitter on a shared machine.
    p99 = min(_p99_compute(frames_right)[0] for _ in range(3))

    print(f"\n[loop-rate] p99 compute={p99:.0f} µs  "
          f"(period={period:.0f}, baseline={base:.0f}, margin={margin})")

    # (a) ABSOLUTE real-time guarantee.
    assert p99 < period, (
        f"p99 compute {p99:.0f} µs exceeds the {period:.0f} µs frame — the loop is "
        "NOT real-time. Stop; report (PD/filter cost).")

    # (b) REGRESSION guard.
    cap = base * (1.0 + margin)
    assert p99 < cap, (
        f"p99 compute {p99:.0f} µs > baseline {base:.0f} * (1+{margin}) = {cap:.0f} µs. "
        "A silent slowdown crept in (still under the ceiling). Investigate before "
        "bumping the baseline.")
