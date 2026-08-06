"""Deterministic grasp/contact scenarios for the G2 dynamic gate.

Two tiny, fixed objects (ticket Build §3): a CYLINDER (power grasp) and a SPHERE
(power-enveloping grasp — the intended fingertip *pinch* is reported as a tuning
finding, see below). Each is driven by a slow, ramped closing command (a safe
teleop trajectory closes over many frames; slamming the fingers shut launches the
object — an ejection artifact, not a grasp). Objects rest against the PALM, held by
a mild palm-ward gravity (there is no arm/wrist to hold against world gravity), and
the fingers + thumb cage them against that palm.

TUNING FINDINGS (ADR-0009; ticket: "report tuning findings, don't force a green"):
  * A free fingertip PINCH is fragile: the thumb's opposition sweep is lateral and
    bats a small free sphere out of the gap before the finger closes (no wrist to
    pre-load the pinch). The robust, deterministic contact scenario for the sphere
    is therefore a palm-backed enveloping grasp, not a 2-point fingertip pinch.
  * A per-joint torque cap does NOT linearly bound total grip force (fingers sum on
    one object); the cap (``PDGains.max_force_nm``) is tuned to the WORST observed
    grasp so measured contact force stays <= the 15 N safety cap.

Geometry is finger-curl-derived (palm faces +x; fingers curl +x; the closed "C"
pocket centroid is ~(0.017, 0, 0.17)). All poses/seeds are fixed -> reproducible.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pybullet as pb

from .conventions import N_JOINTS
from .dynamics import L20Dynamics, PDGains, DEFAULT_GAINS

# Palm-ward gravity: presses the object onto the palm's +x face so the fingers have
# a reaction surface to cage against (there is no arm to resist world gravity).
PALM_GRAVITY = (-4.0, 0.0, 0.0)

# Closing command (semantic idx -> rad): all four fingers flex + curl, thumb opposes
# and flexes. Reached by a slow ramp from the open hand.
POWER_CLOSE = {1: 1.2, 2: 1.2, 3: 1.2, 4: 1.2,        # base flex (index..little)
               16: 1.3, 17: 1.3, 18: 1.3, 19: 1.3,    # tip (distal) curl
               10: 0.9, 0: 0.6, 15: 0.7, 5: 0.2}      # thumb opp/base/tip/abd


@dataclass(frozen=True)
class GraspScenario:
    name: str
    kind: str               # 'cylinder' | 'sphere'
    radius: float
    position: tuple         # base position of the object (palm-backed)
    length: float = 0.10    # cylinder only
    mass: float = 0.02
    friction: float = 1.4
    close: dict = field(default_factory=lambda: dict(POWER_CLOSE))


# Tuned, reproducible scenarios (see grasp tuning in the handoff / ADR-0009).
CYLINDER = GraspScenario(
    name="cylinder_power", kind="cylinder", radius=0.016,
    position=(0.055, 0.0, 0.12), length=0.10, mass=0.03, friction=1.2)
SPHERE = GraspScenario(
    name="sphere_envelop", kind="sphere", radius=0.018,
    position=(0.055, 0.0, 0.11), mass=0.02, friction=1.5)
SCENARIOS = {s.name: s for s in (CYLINDER, SPHERE)}


@dataclass
class GraspResult:
    name: str
    settled_pos: np.ndarray       # object pose after pre-grasp settle (pre-close)
    final_pos: np.ndarray
    displacement: float           # |final - settled|, m  (ejection guard)
    end_speed: float              # m/s at the end of the hold (stability)
    n_contact_links: int          # distinct hand links that touched the object
    contact_links: set
    steady_force_N: float         # mean peak contact force over the hold tail
    peak_force_N: float           # max contact force over the hold
    finite: bool                  # object pose/vel stayed finite (no explosion)


def run_grasp(scenario: GraspScenario, side: str = "right",
              gains: PDGains = DEFAULT_GAINS, ramp_steps: int = 600,
              hold_steps: int = 600, settle_steps: int = 120,
              gravity=PALM_GRAVITY, sim_hz: float = 240.0) -> GraspResult:
    """Run one grasp scenario end to end and return contact/stability metrics.

    Sequence: spawn object -> settle onto palm -> RAMP the close command over
    ``ramp_steps`` -> HOLD ``hold_steps``, recording contact force and links.
    """
    dyn = L20Dynamics(side, timestep=1.0 / sim_hz, gains=gains, gravity=gravity)
    try:
        if scenario.kind == "cylinder":
            q = pb.getQuaternionFromEuler([np.pi / 2, 0, 0])  # cylinder axis -> Y
            dyn.add_cylinder(scenario.name, scenario.radius, scenario.length,
                             list(scenario.position), mass=scenario.mass,
                             orientation=q, lateral_friction=scenario.friction)
        else:
            dyn.add_sphere(scenario.name, scenario.radius, list(scenario.position),
                           mass=scenario.mass, lateral_friction=scenario.friction)

        dyn.set_command(np.zeros(N_JOINTS))
        for _ in range(settle_steps):
            dyn.step()
        settled, _ = dyn.object_pose(scenario.name)

        target = np.zeros(N_JOINTS)
        for idx, val in scenario.close.items():
            target[idx] = val
        for k in range(ramp_steps):
            dyn.set_command(((k + 1) / ramp_steps) * target)
            dyn.step()

        forces, links = [], set()
        for _ in range(hold_steps):
            dyn.step()
            forces.append(dyn.max_contact_force(scenario.name))
            links |= dyn.contacting_links(scenario.name)

        final, _ = dyn.object_pose(scenario.name)
        lin, ang = dyn.object_velocity(scenario.name)
        names = {v: k for k, v in dyn.lidx.items()}
        forces = np.asarray(forces)
        finite = bool(np.all(np.isfinite(final)) and np.all(np.isfinite(lin)))
        tail = forces[-150:] if len(forces) >= 150 else forces
        return GraspResult(
            name=scenario.name, settled_pos=settled, final_pos=final,
            displacement=float(np.linalg.norm(final - settled)),
            end_speed=float(np.linalg.norm(lin)),
            n_contact_links=len(links),
            contact_links={names.get(i, i) for i in links},
            steady_force_N=float(tail.mean()) if len(tail) else 0.0,
            peak_force_N=float(forces.max()) if len(forces) else 0.0,
            finite=finite)
    finally:
        dyn.close()
