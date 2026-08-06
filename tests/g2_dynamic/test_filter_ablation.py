"""G2 Test 4 — filter ablation (paper analog): the safety filter is LOAD-BEARING.

Adversarial self-collision command scripts are commanded under dynamics with the
safety filter ON vs OFF. With the filter ON the commanded/achieved config is
self-collision-free; with it OFF the same script penetrates. Penetration is judged
by the project's collision AUTHORITY (``src/safety.CollisionModel``, the analytic
capsule + palm model) — PyBullet's own self-collision is intentionally left off so
the model is the single arbiter (and so an OFF run actually realises the penetration
the filter is meant to prevent).

This proves the filter removes penetrations that would otherwise reach the hand —
the same claim the paper's ablation makes.
"""
import numpy as np
import pytest

from src.sim import L20Dynamics
from src.sim.conventions import N_JOINTS
from src import safety
from src.safety import CollisionModel
from src.safety.config import DEFAULT_CONFIG

from conftest import max_pen_depth

SIDES = ("right", "left")
MARGIN = DEFAULT_CONFIG.separation_margin_m


def _adversarial_scripts():
    """In-limits configs that self-collide per the model (so OFF is a real test)."""
    # closing fist: adjacent distal phalanges overlap (abductions at 0, all in-limits).
    fist = np.zeros(N_JOINTS)
    fist[[1, 2, 3, 4]] = 1.0
    fist[[16, 17, 18, 19]] = 1.2
    fist[0] = 0.6
    fist[10] = 0.5
    fist[15] = 0.7
    # thumb driven across into the index column.
    thumb = np.zeros(N_JOINTS)
    thumb[10] = 1.2
    thumb[0] = 1.0
    thumb[15] = 1.0
    thumb[1] = 1.0
    thumb[16] = 1.2
    return {"closing_fist": fist, "thumb_cross": thumb}


def _depth(model, q):
    return max_pen_depth(model, q, MARGIN)


@pytest.mark.parametrize("side", SIDES)
def test_filter_ablation_load_bearing(side):
    model = CollisionModel(side)
    scripts = _adversarial_scripts()

    # a script counts as adversarial only if it REALLY overlaps (depth > margin),
    # else the OFF arm is vacuous.
    colliding = {name: q for name, q in scripts.items() if _depth(model, q) > MARGIN}
    assert colliding, (
        f"[{side}] no adversarial script truly overlaps the model — cannot prove the "
        "filter is load-bearing; strengthen the scripts.")

    for name, q in colliding.items():
        cand_depth = _depth(model, q)

        # FILTER ON: project, command under dynamics, read achieved -> must be clear.
        safe = safety.filter(q.tolist(), None, 1 / 30.0, side=side)
        on_cmd_depth = _depth(model, safe["joint_rad"])
        dyn = L20Dynamics(side, gravity=(0, 0, 0))
        try:
            dyn.set_command(safe["joint_rad"])
            for _ in range(200):
                dyn.step()
            on_ach_depth = _depth(model, dyn.achieved_joint_rad())
        finally:
            dyn.close()

        # FILTER OFF: command the raw adversarial script -> overlap realised.
        dyn = L20Dynamics(side, gravity=(0, 0, 0))
        try:
            dyn.set_command(q.tolist())
            for _ in range(200):
                dyn.step()
            off_ach_depth = _depth(model, dyn.achieved_joint_rad())
        finally:
            dyn.close()

        print(f"\n[ablation/{side}] {name} (mm): candidate={cand_depth*1e3:.2f} | "
              f"ON cmd={on_cmd_depth*1e3:.2f} ach={on_ach_depth*1e3:.2f} | "
              f"OFF ach={off_ach_depth*1e3:.2f}  (margin {MARGIN*1e3:.1f})")

        assert safe["modified"], f"[{side}] {name}: filter left a colliding cmd unmodified"
        assert on_cmd_depth <= MARGIN + 1e-6, (
            f"[{side}] {name}: filtered command still overlaps {on_cmd_depth*1e3:.2f} mm")
        assert on_ach_depth <= MARGIN + 1e-6, (
            f"[{side}] {name}: filter ON but achieved config overlaps "
            f"{on_ach_depth*1e3:.2f} mm")
        assert off_ach_depth > MARGIN, (
            f"[{side}] {name}: filter OFF did NOT overlap ({off_ach_depth*1e3:.2f} mm) — "
            "ablation is vacuous (filter not demonstrably load-bearing)")
