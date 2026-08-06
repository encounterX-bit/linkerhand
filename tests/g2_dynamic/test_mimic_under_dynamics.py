"""G2 Test 2 — mimic coupling holds UNDER DYNAMICS.

PyBullet ignores URDF <mimic>; the harness re-issues each mimic setpoint at
``ratio*driver + offset`` every physics step. This verifies the coupling ratios
(0.8917 non-thumb / 1.1619 thumb) hold while STEPPING (a ramp + a settled hold),
not just statically — harder, because the mimic motor must track the moving driver.
"""
import numpy as np
import pytest

from src.sim import L20Dynamics
from src.sim.conventions import N_JOINTS

SIDES = ("right", "left")


def _ramp_to(dyn, target, ramp=400, hold=200):
    target = np.asarray(target, float)
    for k in range(ramp):
        dyn.set_command(((k + 1) / ramp) * target)
        dyn.step()
    for _ in range(hold):
        dyn.step()


@pytest.mark.parametrize("side", SIDES)
def test_mimic_abs_error_under_dynamics(side, baseline):
    """abs(actual - ratio*driver) stays within tolerance through a ramp + hold."""
    tol = baseline["mimic"]["abs_err_tol_rad"]
    dyn = L20Dynamics(side)
    try:
        # close all fingers + thumb so every mimic is exercised away from zero.
        target = np.zeros(N_JOINTS)
        for i in (1, 2, 3, 4):
            target[i] = 0.8
        for i in (16, 17, 18, 19):
            target[i] = 0.9
        target[15] = 0.8

        worst_during_ramp = 0.0
        target = np.asarray(target)
        for k in range(400):
            dyn.set_command(((k + 1) / 400) * target)
            dyn.step()
            worst_during_ramp = max(
                worst_during_ramp,
                max(v[2] for v in dyn.mimic_residuals().values()))
        for _ in range(200):
            dyn.step()
        worst_settled = max(v[2] for v in dyn.mimic_residuals().values())

        print(f"\n[mimic/{side}] worst abs-err during ramp={worst_during_ramp:.4f} "
              f"settled={worst_settled:.4f} rad (tol {tol})")
        assert worst_settled <= tol, (
            f"settled mimic abs error {worst_settled:.4f} > {tol} rad ({side})")
        # ramp transient is allowed a little more head-room (motor lag) but must
        # not blow up — bound it generously.
        assert worst_during_ramp <= tol * 3, (
            f"mimic abs error during ramp {worst_during_ramp:.4f} rad too large ({side})")
    finally:
        dyn.close()


@pytest.mark.parametrize("side", SIDES)
def test_mimic_ratio_under_dynamics(side, baseline):
    """Realized ratio (mimic/driver) at a flexed hold matches the nominal ratio."""
    b = baseline["mimic"]
    rel = b["ratio_rel_tol"]
    dyn = L20Dynamics(side)
    try:
        target = np.zeros(N_JOINTS)
        for i in (16, 17, 18, 19):
            target[i] = 0.9      # non-thumb pip drivers
        target[15] = 0.8         # thumb_mcp driver
        _ramp_to(dyn, target)

        for mname, (driver, mult, off) in dyn.mimics.items():
            a = dyn.driver_angle(driver)
            m = dyn.joint_state(mname)[0]
            assert abs(a) > 0.3, f"{mname} driver not flexed enough to test ratio"
            realized = m / a
            nominal = b["ratio_thumb"] if "thumb" in mname else b["ratio_nonthumb"]
            # nominal_mult from URDF should equal our baseline nominal
            assert abs(mult - nominal) < 1e-3, (
                f"{mname} URDF mult {mult} != baseline nominal {nominal}")
            rel_err = abs(realized - nominal) / nominal
            print(f"[mimic-ratio/{side}] {mname}: realized {realized:.4f} "
                  f"nominal {nominal:.4f} rel_err {rel_err:.3f}")
            assert rel_err <= rel, (
                f"{mname} realized ratio {realized:.4f} off nominal {nominal:.4f} "
                f"by {rel_err:.3f} > {rel} ({side})")
    finally:
        dyn.close()
