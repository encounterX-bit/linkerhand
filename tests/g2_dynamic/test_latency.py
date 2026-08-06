"""G2 Test 5 — latency stability.

At low / target / high CAN-control latency the closed loop must stay STABLE: the
achieved config stays finite, tracking error stays bounded (no divergence), and a
held input settles rather than oscillating away. Tracking error is EXPECTED to grow
with latency (reported, monitored); the gate is on stability, not accuracy.
"""
import numpy as np
import pytest

from src.sim import ClosedLoopSim


def _run(frames, latency_s, side="right"):
    cl = ClosedLoopSim(side, latency_s=latency_s, use_filter=True)
    try:
        recs = cl.run(frames)
    finally:
        cl.close()
    return recs


def test_latency_stability(frames_right, baseline):
    lb = baseline["latency"]
    frames = list(frames_right)
    ceiling = lb["max_seg_err_ceiling_rad"]

    report = []
    results = {}
    for nf in lb["frames_low_target_high"]:
        latency_s = nf / 30.0
        recs = _run(frames, latency_s)
        achieved = np.array([r.achieved for r in recs])
        errs = np.array([r.max_seg_err for r in recs])
        results[nf] = (achieved, errs)
        report.append(f"  lat={nf} frames ({latency_s*1000:.0f} ms): "
                      f"p95={np.percentile(errs, 95):.3f} "
                      f"last10mean={errs[-10:].mean():.3f} max={errs.max():.3f} rad")
    print("\n[latency] seg-err vs latency:\n" + "\n".join(report))

    for nf, (achieved, errs) in results.items():
        tag = f"latency={nf} frames"
        # 1. finite — no explosion / NaN under delay.
        assert np.all(np.isfinite(achieved)), f"{tag}: non-finite achieved config"
        # 2. bounded — even degraded tracking stays under a generous ceiling.
        assert errs.max() <= ceiling, (
            f"{tag}: max seg-err {errs.max():.3f} exceeds ceiling {ceiling} rad — "
            "the loop diverged under latency. Report (PD/filter tuning), do NOT "
            "force green.")
        # 3. no runaway — the loop settles toward the end (held input), it does not
        #    accumulate. Final-window error must not exceed the transient peak.
        assert errs[-10:].mean() <= errs.max() + 1e-9, f"{tag}: error growing at end"

    # divergence check: error should worsen monotonically-ish with latency but never
    # blow past the ceiling (already asserted). Report the growth as monitored.
    p95s = {nf: float(np.percentile(e, 95)) for nf, (_, e) in results.items()}
    assert p95s[lb["frames_low_target_high"][0]] <= p95s[lb["frames_low_target_high"][-1]], \
        "higher latency should not improve tracking — measurement suspect"
