"""G2 Test 3 — grasp / contact + virtual force cap.

Each tiny object (cylinder = power grasp; sphere = enveloping grasp) is closed on by
a slow ramped command. We assert the safety-relevant properties:

  * contact established (>= min_contact_links distinct hand links, incl. the thumb),
  * NOT ejected / exploded — bounded displacement + end speed, all finite,
  * contact force <= the virtual cap (the 15 N ForceClampSpec, the safety law).

Plus a load-bearing check that the cap actually BINDS: with the motor torque raised
well above the tuned cap, the sphere grasp force exceeds 15 N — so the tuned cap is
what holds force down, not light objects.

Tuning findings (fragile fingertip pinch; per-joint torque != total grip force) are
recorded in src/sim/grasp.py + ADR-0009 and were NOT forced green.
"""
import json
import os

import numpy as np
import pytest

from src.sim import run_grasp, CYLINDER, SPHERE, PDGains
from src.safety import DEFAULT_CONFIG

from conftest import ensure_out

SCENARIOS = [CYLINDER, SPHERE]
SAFETY_FORCE_CAP_N = DEFAULT_CONFIG.force.max_grip_force_N  # 15 N (the real law)


@pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda s: s.name)
def test_grasp_stable_and_capped(scenario, baseline):
    b = baseline["grasp"]
    assert abs(SAFETY_FORCE_CAP_N - b["force_cap_N"]) < 1e-9, \
        "baseline force cap drifted from src/safety ForceClampSpec"

    r = run_grasp(scenario)
    summary = (f"{r.name}: disp={r.displacement:.3f} m end_speed={r.end_speed:.4f} "
               f"links={r.n_contact_links} steady={r.steady_force_N:.2f} N "
               f"peak={r.peak_force_N:.2f} N finite={r.finite}\n  "
               f"contacts={sorted(str(x) for x in r.contact_links)}")
    print("\n[grasp] " + summary)

    # not exploded
    assert r.finite, f"object pose/vel went non-finite — exploded\n{summary}"
    # not ejected (bounded motion)
    assert r.displacement <= b["max_displacement_m"], (
        f"object displaced {r.displacement:.3f} m > {b['max_displacement_m']} — "
        f"ejected, not grasped\n{summary}")
    assert r.end_speed <= b["max_end_speed_m_s"], (
        f"object still moving {r.end_speed:.4f} m/s — not a settled grasp\n{summary}")
    # contact established
    assert r.n_contact_links >= b["min_contact_links"], (
        f"only {r.n_contact_links} contact link(s); expected >= "
        f"{b['min_contact_links']}\n{summary}")
    # FORCE CAP (the safety law): measured contact force <= cap.
    assert r.peak_force_N <= SAFETY_FORCE_CAP_N, (
        f"peak contact force {r.peak_force_N:.2f} N exceeds the {SAFETY_FORCE_CAP_N} N "
        f"virtual cap — tune PDGains.max_force_nm down\n{summary}")


def test_force_cap_is_load_bearing():
    """Raising the motor torque well above the tuned cap pushes grip force OVER 15 N
    — proving the tuned cap (not light objects) is what holds force under the law."""
    capped = run_grasp(SPHERE)                       # default tuned cap (0.12 Nm)
    uncapped = run_grasp(SPHERE, gains=PDGains(max_force_nm=0.5, mimic_max_force_nm=0.7))
    print(f"\n[force-cap binds] capped peak={capped.peak_force_N:.2f} N  "
          f"uncapped peak={uncapped.peak_force_N:.2f} N  (cap {SAFETY_FORCE_CAP_N} N)")
    assert capped.peak_force_N <= SAFETY_FORCE_CAP_N, "tuned cap should hold <= 15 N"
    assert uncapped.peak_force_N > SAFETY_FORCE_CAP_N, (
        "raising torque did NOT exceed the cap — then the cap is not what bounds "
        "force, and the grasp test would pass vacuously")


def test_write_grasp_demo_artifact(baseline):
    """Closed-loop grasp demo artifact (ticket 'Done =' deliverable)."""
    out = ensure_out()
    rows = []
    for scenario in SCENARIOS:
        r = run_grasp(scenario)
        rows.append({
            "scenario": r.name, "kind": scenario.kind,
            "object_radius_m": scenario.radius,
            "settled_pos": [float(x) for x in r.settled_pos],
            "final_pos": [float(x) for x in r.final_pos],
            "displacement_m": r.displacement, "end_speed_m_s": r.end_speed,
            "n_contact_links": r.n_contact_links,
            "contact_links": sorted(str(x) for x in r.contact_links),
            "steady_force_N": r.steady_force_N, "peak_force_N": r.peak_force_N,
            "force_cap_N": SAFETY_FORCE_CAP_N, "finite": r.finite,
        })
    path = os.path.join(out, "grasp_demo.json")
    with open(path, "w") as f:
        json.dump({"force_cap_N": SAFETY_FORCE_CAP_N, "grasps": rows}, f, indent=2)
    assert os.path.exists(path)
    print(f"\n[grasp-demo] wrote {path}")
