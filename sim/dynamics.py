"""L20 DYNAMIC harness on PyBullet (gate G2) — masses, gravity, contact, motors.

This is the G2 promotion of the G1 kinematic harness (``kinematics.py``). Where G1
*placed* joints with ``resetJointState`` (no physics), G2 *drives* them with
position-PD motors (``setJointMotorControl2``) under gravity and contact, steps a
real simulation, and reads back the achieved state.

Responsibilities:
  1. **Dynamics.** Loads the L20 URDF with its URDF masses/inertias, fixed base,
     gravity on. The 16 active DoF are position-PD controlled; contact + friction
     are PyBullet's.
  2. **Mimic enforcement UNDER DYNAMICS.** PyBullet ignores ``<mimic>``. Each step we
     re-issue a POSITION_CONTROL setpoint for every mimic joint at
     ``ratio*driver_cmd + offset`` (per-step enforcement; the ticket's sanctioned
     alternative to a gear constraint). Ratios are verified to hold while stepping
     (``test_mimic_under_dynamics``).
  3. **Metric FK is NOT PyBullet.** Retargeting *quality* (segment directions) is
     always measured by the ONE FK authority ``src/kinematics`` at the *achieved*
     joint vector — PyBullet provides dynamics/contact only (ADR-0005).

PD GAINS ARE SIM-ONLY. They tune the contact simulation; they are NOT hardware
control gains and never leave the sim. Force is capped virtually here; the REAL
clamp is comms/G3 (``src/safety`` force spec, ``HW_ENABLE_TOKEN``).
"""
from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

import numpy as np
import pybullet as pb

from src.kinematics import L20FK

from .conventions import ACTIVE_IDX, RESERVED_IDX, JOINT_NAME, BASE_LINK, N_JOINTS
from .kinematics import urdf_path, _parse_mimics


@dataclass(frozen=True)
class PDGains:
    """SIM-ONLY position-PD gains + virtual torque cap. NOT hardware gains.

    ``max_force_nm`` is the per-joint motor torque ceiling — the *virtual force cap*.
    IMPORTANT (tuning finding, ADR-0009): a per-joint torque limit does NOT linearly
    bound the TOTAL grip force, because several fingers (and the thumb) sum their
    contributions onto one object and a point/line contact concentrates them. At
    0.35 Nm a sphere power grasp reached ~24-28 N (> the 15 N cap); 0.12 Nm holds
    both the cylinder (peak ~12.5 N) and the sphere (peak ~9 N) UNDER the 15 N
    ``ForceClampSpec`` cap while still tracking free motion to ~0.01 rad. So 0.12 Nm
    is the empirically-chosen virtual cap for the worst observed grasp; raising it
    breaks the cap (which is what makes ``test_grasp_force_cap`` load-bearing). The
    REAL force clamp is comms/G3, not this.

    ``kp``/``kd`` are PyBullet POSITION_CONTROL position/velocity gains.
    """

    kp: float = 0.3
    kd: float = 0.6
    max_force_nm: float = 0.12          # virtual grip cap: worst grasp <= 15 N
    mimic_max_force_nm: float = 0.17     # coupled distal follows the driver


DEFAULT_GAINS = PDGains()


class L20Dynamics:
    """Dynamic L20 model for one side ('right'/'left').

    Owns a private PyBullet DIRECT client with gravity + contact. Call ``close``
    (or use as a context manager) to release it.
    """

    def __init__(self, side: str = "right", timestep: float = 1.0 / 240.0,
                 gains: PDGains = DEFAULT_GAINS, gravity=(0.0, 0.0, -9.81),
                 substeps: int = 1, gui: bool = False):
        if side not in BASE_LINK:
            raise ValueError(f"side must be 'right'/'left', got {side!r}")
        self.side = side
        self.path = urdf_path(side)
        self.gains = gains
        self.timestep = float(timestep)
        self.substeps = int(substeps)
        self._fk = L20FK(side)                 # metric FK authority (NOT dynamics)
        self._cmd = np.zeros(N_JOINTS)         # last commanded 20-vector

        self.cid = pb.connect(pb.GUI if gui else pb.DIRECT)
        if gui:
            pb.configureDebugVisualizer(pb.COV_ENABLE_GUI, 0,
                                        physicsClientId=self.cid)
            pb.resetDebugVisualizerCamera(
                cameraDistance=0.38, cameraYaw=50, cameraPitch=-28,
                cameraTargetPosition=[0.035, 0.0, 0.10],
                physicsClientId=self.cid)
        pb.setGravity(*gravity, physicsClientId=self.cid)
        pb.setPhysicsEngineParameter(fixedTimeStep=self.timestep,
                                     numSubSteps=self.substeps,
                                     physicsClientId=self.cid)
        self.body = pb.loadURDF(self.path, useFixedBase=True,
                                physicsClientId=self.cid)

        self.jidx, self.lidx, self._limit = {}, {}, {}
        for i in range(pb.getNumJoints(self.body, physicsClientId=self.cid)):
            info = pb.getJointInfo(self.body, i, physicsClientId=self.cid)
            jname = info[1].decode()
            self.jidx[jname] = i
            self.lidx[info[12].decode()] = i
            self._limit[jname] = (float(info[8]), float(info[9]))
        self.mimics = _parse_mimics(self.path)

        # Disable the default velocity motors so our POSITION_CONTROL is the only
        # actuator (PyBullet adds a stiff velocity motor by default).
        for i in self.jidx.values():
            pb.setJointMotorControl2(self.body, i, pb.VELOCITY_CONTROL, force=0.0,
                                     physicsClientId=self.cid)
        self.objects = {}                      # name -> body id (graspable props)
        self.set_command(np.zeros(N_JOINTS))   # open hand, hold
        # settle to the open pose so step 0 of a loop starts from rest
        self._apply_motors()

    # -- context manager -------------------------------------------------- #
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        if getattr(self, "cid", None) is not None:
            try:
                pb.disconnect(physicsClientId=self.cid)
            finally:
                self.cid = None

    # -- limits ----------------------------------------------------------- #
    def active_limits(self):
        return {idx: self._limit[JOINT_NAME[idx]] for idx in ACTIVE_IDX}

    # -- command / motors ------------------------------------------------- #
    def set_command(self, joint_rad):
        """Set the position-PD targets for the 16 active DoF + the 5 mimics.

        Stores the 20-vector command (reserved idx forced to 0.0) and computes each
        mimic setpoint as ``ratio*driver + offset``. Call ``step`` to advance physics.
        """
        if len(joint_rad) != N_JOINTS:
            raise ValueError(f"joint_rad must have {N_JOINTS} entries")
        cmd = np.asarray(joint_rad, dtype=float).copy()
        cmd[list(RESERVED_IDX)] = 0.0
        self._cmd = cmd

    def _apply_motors(self):
        g = self.gains
        for idx in ACTIVE_IDX:
            jname = JOINT_NAME[idx]
            pb.setJointMotorControl2(
                self.body, self.jidx[jname], pb.POSITION_CONTROL,
                targetPosition=float(self._cmd[idx]),
                positionGain=g.kp, velocityGain=g.kd, force=g.max_force_nm,
                physicsClientId=self.cid)
        # MIMIC ENFORCEMENT UNDER DYNAMICS (re-issued every step).
        applied = {JOINT_NAME[idx]: float(self._cmd[idx]) for idx in ACTIVE_IDX}
        for mname, (driver, mult, off) in self.mimics.items():
            tgt = mult * applied[driver] + off
            pb.setJointMotorControl2(
                self.body, self.jidx[mname], pb.POSITION_CONTROL,
                targetPosition=tgt, positionGain=g.kp, velocityGain=g.kd,
                force=g.mimic_max_force_nm, physicsClientId=self.cid)

    def step(self, n: int = 1):
        """Advance the dynamics ``n`` steps, re-enforcing the mimics each step."""
        for _ in range(n):
            self._apply_motors()
            pb.stepSimulation(physicsClientId=self.cid)

    # -- read-back -------------------------------------------------------- #
    def joint_state(self, jname):
        return pb.getJointState(self.body, self.jidx[jname],
                                physicsClientId=self.cid)

    def achieved_joint_rad(self):
        """The achieved 20-vector (active drivers from physics; reserved = 0)."""
        q = np.zeros(N_JOINTS)
        for idx in ACTIVE_IDX:
            q[idx] = self.joint_state(JOINT_NAME[idx])[0]
        return q

    def driver_angle(self, jname):
        return self.joint_state(jname)[0]

    def mimic_residuals(self):
        """{mimic_joint: (actual, expected, abs_err)} for the achieved state.

        ``expected = ratio*driver_actual + offset`` (the coupling that must hold
        while stepping). The abs error is what ``test_mimic_under_dynamics`` bounds.
        """
        out = {}
        for mname, (driver, mult, off) in self.mimics.items():
            actual = self.joint_state(mname)[0]
            expected = mult * self.driver_angle(driver) + off
            out[mname] = (actual, expected, abs(actual - expected))
        return out

    # -- metric FK (authority, NOT pybullet) ------------------------------ #
    def segment_dirs(self):
        """{finger: (r_prox, r_dist)} at the ACHIEVED config, via src/kinematics."""
        return self._fk.all_segment_dirs(self.achieved_joint_rad())

    # -- objects / contact ------------------------------------------------ #
    def add_cylinder(self, name, radius, length, base_pos, mass=0.02,
                     orientation=(0, 0, 0, 1), lateral_friction=1.0):
        col = pb.createCollisionShape(pb.GEOM_CYLINDER, radius=radius, height=length,
                                      physicsClientId=self.cid)
        bid = pb.createMultiBody(baseMass=mass, baseCollisionShapeIndex=col,
                                 basePosition=base_pos, baseOrientation=orientation,
                                 physicsClientId=self.cid)
        pb.changeDynamics(bid, -1, lateralFriction=lateral_friction,
                          physicsClientId=self.cid)
        self.objects[name] = bid
        return bid

    def add_sphere(self, name, radius, base_pos, mass=0.01, lateral_friction=1.0):
        col = pb.createCollisionShape(pb.GEOM_SPHERE, radius=radius,
                                      physicsClientId=self.cid)
        bid = pb.createMultiBody(baseMass=mass, baseCollisionShapeIndex=col,
                                 basePosition=base_pos, physicsClientId=self.cid)
        pb.changeDynamics(bid, -1, lateralFriction=lateral_friction,
                          physicsClientId=self.cid)
        self.objects[name] = bid
        return bid

    def add_box(self, name, half_extents, base_pos, mass=0.02,
                orientation=(0, 0, 0, 1), lateral_friction=1.0,
                rgba=(0.1, 0.45, 0.9, 1.0)):
        col = pb.createCollisionShape(pb.GEOM_BOX, halfExtents=half_extents,
                                      physicsClientId=self.cid)
        vis = pb.createVisualShape(pb.GEOM_BOX, halfExtents=half_extents,
                                   rgbaColor=rgba, physicsClientId=self.cid)
        bid = pb.createMultiBody(baseMass=mass, baseCollisionShapeIndex=col,
                                 baseVisualShapeIndex=vis,
                                 basePosition=base_pos,
                                 baseOrientation=orientation,
                                 physicsClientId=self.cid)
        pb.changeDynamics(bid, -1, lateralFriction=lateral_friction,
                          rollingFriction=0.002, spinningFriction=0.002,
                          physicsClientId=self.cid)
        self.objects[name] = bid
        return bid

    def object_pose(self, name):
        pos, orn = pb.getBasePositionAndOrientation(self.objects[name],
                                                    physicsClientId=self.cid)
        return np.asarray(pos), np.asarray(orn)

    def object_velocity(self, name):
        lin, ang = pb.getBaseVelocity(self.objects[name], physicsClientId=self.cid)
        return np.asarray(lin), np.asarray(ang)

    def contacts_on(self, name):
        """List of (link_index, normal_force) contacts between the hand and object."""
        cps = pb.getContactPoints(bodyA=self.body, bodyB=self.objects[name],
                                  physicsClientId=self.cid)
        return [(cp[3], float(cp[9])) for cp in cps]

    def max_contact_force(self, name):
        cps = self.contacts_on(name)
        return max((f for _, f in cps), default=0.0)

    def contacting_links(self, name):
        return {cp[0] for cp in self.contacts_on(name)}
