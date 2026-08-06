"""G2 Test 8 — closed loop WITH the safety filter, end to end.

Replays the recorded real sequence through the full loop (landmarks -> retarget ->
safety.filter -> command -> step -> read) for both sides and asserts the loop is
well-behaved with the G2 guard inline: every commanded config is contract-clean
(reserved 0, finite), the filter is actually exercised (it modifies at least some
frames), and every applied command is self-collision-free per the safety model.

This is the sync-point test that needs ``safety.filter`` to have landed (it has,
g2_safety 47/47). It writes a per-frame CSV demo artifact.
"""
import csv
import os

import numpy as np
import pytest

from src.sim import ClosedLoopSim
from src.sim.conventions import RESERVED_IDX
from src.safety import CollisionModel
from src.safety.config import DEFAULT_CONFIG

from conftest import ensure_out, max_pen_depth

SIDES = ("right", "left")
MARGIN = DEFAULT_CONFIG.separation_margin_m


@pytest.mark.parametrize("side", SIDES)
def test_closed_loop_with_filter(side, request):
    frames = request.getfixturevalue("frames_right" if side == "right" else "frames_left")
    model = CollisionModel(side)
    cl = ClosedLoopSim(side, latency_s=0.0, use_filter=True)
    try:
        recs = cl.run(frames)
    finally:
        cl.close()

    assert recs, f"{side}: no frames replayed"
    n_modified = sum(1 for r in recs if r.filter_modified)
    worst_depth = 0.0
    for r in recs:
        cmd = np.asarray(r.applied, float)
        assert np.all(np.isfinite(cmd)), f"{side}: non-finite command"
        for idx in RESERVED_IDX:
            assert cmd[idx] == 0.0, f"{side}: reserved idx {idx} != 0"
        worst_depth = max(worst_depth, max_pen_depth(model, cmd, MARGIN))

    print(f"\n[closed-loop+filter/{side}] {len(recs)} frames, "
          f"filter modified {n_modified}, worst applied-cmd overlap depth "
          f"{worst_depth*1000:.3f} mm (margin {MARGIN*1000:.1f} mm)")
    # every APPLIED command is non-penetrating (no real surface overlap). The filter
    # projects to the margin buffer; depth <= margin means the surfaces are clear.
    assert worst_depth <= MARGIN + 1e-6, (
        f"{side}: an applied command overlaps surfaces by {worst_depth*1000:.3f} mm "
        f"(> {MARGIN*1000:.1f} mm margin) — the inline filter is not protecting the "
        "command stream")
    # the guard must actually be doing work on this sequence (not a no-op pass-through)
    assert n_modified > 0, (
        f"{side}: filter never modified any frame — either the sequence needs no "
        "guarding or the filter is not wired in; verify the seam")

    # per-frame CSV demo artifact
    out = ensure_out()
    path = os.path.join(out, f"closed_loop_{side}.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "compute_us", "filter_modified", "filter_reason",
                    "max_seg_err_rad"])
        for r in recs:
            w.writerow([r.frame, f"{r.compute_us:.1f}", int(r.filter_modified),
                        r.filter_reason or "", f"{r.max_seg_err:.4f}"])
    print(f"[closed-loop+filter/{side}] wrote {path}")
