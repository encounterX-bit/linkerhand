"""G1 residual gate — SPLIT into a proximal HARD GATE and a distal MONITOR.

History: this used to be one fail-closed ``test_real_set_residual`` that lumped the
(exact) proximal bones with the (under-actuation-limited) coupled distal. Now that
the metric is trustworthy (Finding 1 fixed; ``r_dist`` fingertip-inclusive) the
ticket (sim-agent-G2-v2 Step 0) splits it so G1 is honestly green:

  * ``test_proximal_residual`` — HARD GATE. Pooled proximal geodesic error p95 must
    be <= ``PROXIMAL_TOL``. This is what makes G1 honestly green and is the entry-gate
    condition for the G2 dynamic work.
  * ``test_distal_residual_monitored`` — MONITOR + REGRESSION GUARD. Reports the
    distal distribution (overall + per-finger p50/p95) and passes by DEFAULT (no
    absolute quality line on *synthetic* data), but HARD-FAILS on regression beyond
    ``REGRESSION_MARGIN`` above the committed baseline. When a human sets
    ``G1_DISTAL_RESIDUAL_THRESHOLD`` (from real-camera data) it also becomes an
    absolute gate.

Both replay the recorded real sequences from ``fixtures/real/`` (perception's
convergence point). Until they land, both SKIP. The committed numbers + thresholds
live in ``residual_baseline.json`` (env-overridable). Artifacts (CSV + overlay GIF)
are written to ``out/`` regardless of pass/fail for the human review.

HUMAN SIGN-OFF (see residual_baseline.json + STATE.md): the proximal ``tol_rad`` and
the distal ``regression_margin`` are PROPOSED, not blessed. NOTE the proximal tail is
NOT near-zero (pooled p95=0.1236, not the p50=0.000 the older framing implied).
"""
import glob
import json
import os

import numpy as np
import pytest

from src.sim import L20Kinematics, FINGER_ORDER
from src.sim.pipeline import track_frame, error_rows
from src.sim import viz

HERE = os.path.dirname(os.path.abspath(__file__))
REAL_DIR = os.path.join(HERE, "fixtures", "real")
OUT_DIR = os.path.join(HERE, "out")
BASELINE_PATH = os.path.join(HERE, "residual_baseline.json")

with open(BASELINE_PATH) as _f:
    BASELINE = json.load(_f)


def _load_frames(path):
    """Yield (side, landmarks, t) for DETECTED frames of a perception fixture."""
    doc = json.load(open(path))
    if isinstance(doc, list):
        seq, side = doc, None
    elif "frames" in doc:
        seq, side = doc["frames"], doc.get("side")
    else:
        seq, side = [doc], doc.get("side")
    for i, fr in enumerate(seq):
        if fr.get("detected", True) is False:
            continue
        yield fr.get("side", side), fr["landmarks"], fr.get("t", float(i))


def _real_files():
    return sorted(glob.glob(os.path.join(REAL_DIR, "*.json")))


# --- replay is shared by both tests; memoize so PyBullet loads once ----------- #
_REPLAY_CACHE = None


def _replay():
    """Replay all real fixtures once; return (records, labels, rows, prox, dist).

    ``prox``/``dist`` are {(finger): [err...]} per-segment pools; the flat ``rows``
    carry both segments for the CSV/overlay artifacts.
    """
    global _REPLAY_CACHE
    if _REPLAY_CACHE is not None:
        return _REPLAY_CACHE
    kins = {}
    records, labels, rows = [], [], []
    prox = {f: [] for f in FINGER_ORDER}
    dist = {f: [] for f in FINGER_ORDER}
    frame_no = 0
    for path in _real_files():
        base = os.path.basename(path)
        for side, lm, t in _load_frames(path):
            kin = kins.get(side) or kins.setdefault(side, L20Kinematics(side))
            rec = track_frame(kin, lm, side)
            records.append(rec)
            labels.append(f"{base}#{frame_no}")
            rows.extend(error_rows(rec, frame=frame_no, t=t))
            for f in FINGER_ORDER:
                prox[f].append(rec["err"][f][0])
                dist[f].append(rec["err"][f][1])
            frame_no += 1
    for k in kins.values():
        k.close()
    _REPLAY_CACHE = (records, labels, rows, prox, dist)
    return _REPLAY_CACHE


def _pool(d):
    return np.array([v for vals in d.values() for v in vals], dtype=float)


def _summary(prox, dist):
    pp, dp = _pool(prox), _pool(dist)
    lines = [
        f"proximal pooled (rad): p50={np.percentile(pp, 50):.4f} "
        f"p90={np.percentile(pp, 90):.4f} p95={np.percentile(pp, 95):.4f} "
        f"p99={np.percentile(pp, 99):.4f} max={pp.max():.4f}",
        f"distal   pooled (rad): p50={np.percentile(dp, 50):.4f} "
        f"p90={np.percentile(dp, 90):.4f} p95={np.percentile(dp, 95):.4f} "
        f"p99={np.percentile(dp, 99):.4f} max={dp.max():.4f}",
    ]
    for f in FINGER_ORDER:
        lines.append(
            f"  {f:7s} prox p95={np.percentile(prox[f], 95):.4f}  "
            f"dist p50={np.percentile(dist[f], 50):.4f} "
            f"p95={np.percentile(dist[f], 95):.4f}")
    return "\n".join(lines)


def _require_fixtures():
    if not _real_files():
        pytest.skip(
            f"perception real fixtures not yet landed ({REAL_DIR}/*.json) "
            "— see ticket §Sequencing")


def test_proximal_residual():
    """HARD GATE: pooled proximal geodesic error p95 <= PROXIMAL_TOL.

    Proximal bones are set by the BASE DoF (exact subproblem on reachable). The tail
    is pose-correlated (mid-curl transition frames) under-actuation, not noise; the
    bound is justified in residual_baseline.json and AWAITS human sign-off.
    """
    _require_fixtures()
    records, labels, rows, prox, dist = _replay()
    assert rows, "real fixtures present but no detected frames replayed"

    # artifacts for the human (written once, regardless of pass/fail)
    viz.write_csv(rows, os.path.join(OUT_DIR, "real_residual.csv"))

    tol = float(os.environ.get("PROXIMAL_TOL", BASELINE["proximal"]["tol_rad"]))
    pp = _pool(prox)
    p95 = float(np.percentile(pp, 95))
    summary = _summary(prox, dist)
    print(f"\n[proximal HARD GATE] pooled p95={p95:.4f} rad vs PROXIMAL_TOL={tol}\n"
          + summary)
    assert p95 <= tol, (
        f"PROXIMAL p95 residual {p95:.4f} rad exceeds PROXIMAL_TOL={tol}. Proximal is "
        "the EXACT base-solve segment — a regression here is a real solver/FK break, "
        f"not under-actuation. Do NOT advance; report to the human.\n{summary}")


def test_distal_residual_monitored():
    """MONITOR + REGRESSION GUARD on the coupled distal (under-actuation-limited).

    Pass by default (no absolute quality line on synthetic data); hard-fail only on
    regression beyond REGRESSION_MARGIN above the committed baseline. If a human sets
    G1_DISTAL_RESIDUAL_THRESHOLD it ALSO becomes an absolute gate on distal pooled p95.
    """
    _require_fixtures()
    records, labels, rows, prox, dist = _replay()
    assert rows, "real fixtures present but no detected frames replayed"

    viz.write_csv(rows, os.path.join(OUT_DIR, "real_residual.csv"))
    viz.render_sequence(records, os.path.join(OUT_DIR, "real_sequence"), labels)

    db = BASELINE["distal"]
    margin = float(os.environ.get("REGRESSION_MARGIN", db["regression_margin"]))

    # overall (prox+dist) p95 and thumb distal p95 are the two committed reference
    # numbers (ticket §Step 0). Report the full distal distribution alongside.
    all_err = np.array([r["err_rad"] for r in rows], dtype=float)
    overall_p95 = float(np.percentile(all_err, 95))
    dp = _pool(dist)
    dist_p95 = float(np.percentile(dp, 95))
    thumb_dist_p95 = float(np.percentile(dist["thumb"], 95))
    summary = _summary(prox, dist)
    print(f"\n[distal MONITOR] overall p95={overall_p95:.4f} "
          f"(baseline {db['overall_p95_baseline']}); distal pooled p95={dist_p95:.4f}; "
          f"thumb dist p95={thumb_dist_p95:.4f} (baseline {db['thumb_dist_p95_baseline']}); "
          f"regression_margin={margin}\n" + summary)

    # REGRESSION GUARD (hard fail): catch a real solver/FK regression that inflates
    # the distal residual, while tolerating synthetic re-measure jitter.
    over_cap = db["overall_p95_baseline"] * (1.0 + margin)
    thumb_cap = db["thumb_dist_p95_baseline"] * (1.0 + margin)
    assert overall_p95 <= over_cap, (
        f"REGRESSION: overall p95 {overall_p95:.4f} > baseline "
        f"{db['overall_p95_baseline']} * (1+{margin}) = {over_cap:.4f}. Distal residual "
        f"regressed materially — stop and report.\n{summary}")
    assert thumb_dist_p95 <= thumb_cap, (
        f"REGRESSION: thumb distal p95 {thumb_dist_p95:.4f} > baseline "
        f"{db['thumb_dist_p95_baseline']} * (1+{margin}) = {thumb_cap:.4f}.\n{summary}")

    # OPTIONAL absolute gate once real-camera data exists.
    abs_thr = os.environ.get("G1_DISTAL_RESIDUAL_THRESHOLD")
    if abs_thr is not None:
        abs_thr = float(abs_thr)
        assert dist_p95 <= abs_thr, (
            f"distal pooled p95 {dist_p95:.4f} exceeds "
            f"G1_DISTAL_RESIDUAL_THRESHOLD={abs_thr}.\n{summary}")
