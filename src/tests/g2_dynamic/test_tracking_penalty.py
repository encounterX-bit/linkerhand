"""G2 Test 7 — tracking penalty (MONITORED, not a hard gate).

Quantifies the per-segment orientation error of the FULL dynamic loop (dynamics +
filter + optional latency) against the G1 KINEMATIC numbers (overall p95 ~0.178).
Dynamics lag + the safety projection + transport delay all add error; this test
reports the penalty and writes an artifact. It only HARD-fails if the achieved
config goes non-finite (a real break), per the ticket ("monitored, not a hard gate").
"""
import json
import os

import numpy as np

from src.sim import ClosedLoopSim
from src.sim.conventions import FINGER_ORDER

from conftest import ensure_out


def _seg_stats(recs):
    prox = np.array([[r.seg_err[f][0] for f in FINGER_ORDER] for r in recs]).ravel()
    dist = np.array([[r.seg_err[f][1] for f in FINGER_ORDER] for r in recs]).ravel()
    allv = np.concatenate([prox, dist])
    return {
        "overall_p50": float(np.percentile(allv, 50)),
        "overall_p95": float(np.percentile(allv, 95)),
        "prox_p95": float(np.percentile(prox, 95)),
        "dist_p95": float(np.percentile(dist, 95)),
    }


def test_tracking_penalty_monitored(frames_right, baseline):
    frames = list(frames_right)
    g1_ref = baseline["tracking_penalty"]["g1_kinematic_overall_p95_rad"]

    report = {}
    for tag, kw in (("dyn_nolatency", dict(latency_s=0.0)),
                    ("dyn_target_latency", dict(latency_s=2 / 30.0))):
        cl = ClosedLoopSim("right", use_filter=True, **kw)
        try:
            recs = cl.run(frames)
        finally:
            cl.close()
        stats = _seg_stats(recs)
        assert np.all(np.isfinite([r.max_seg_err for r in recs])), \
            f"{tag}: non-finite tracking error — a real break"
        report[tag] = stats

    report["g1_kinematic_overall_p95"] = g1_ref
    report["penalty_overall_p95_vs_g1"] = (
        report["dyn_nolatency"]["overall_p95"] - g1_ref)

    print("\n[tracking-penalty] G1 kinematic overall p95={:.3f}".format(g1_ref))
    for tag in ("dyn_nolatency", "dyn_target_latency"):
        s = report[tag]
        print(f"  {tag}: overall p50={s['overall_p50']:.3f} p95={s['overall_p95']:.3f} "
              f"(prox {s['prox_p95']:.3f} / dist {s['dist_p95']:.3f})")
    print(f"  penalty (dyn no-latency overall p95 - G1) = "
          f"{report['penalty_overall_p95_vs_g1']:+.3f} rad")

    out = ensure_out()
    with open(os.path.join(out, "tracking_penalty.json"), "w") as f:
        json.dump(report, f, indent=2)
