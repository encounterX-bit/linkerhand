"""L20 kinematic forward-kinematics harness on PyBullet (gate G1).

KINEMATIC ONLY. Joints are placed with ``resetJointState``; there is NO motor
control, NO gravity, NO contact, NO dynamics. This measures retargeting *quality*
(achieved vs target finger-segment directions), not control.

Two responsibilities the rest of the harness leans on:

  1. **Manual mimic enforcement.** PyBullet IGNORES URDF ``<mimic>`` tags. After the
     16 active joints are set, each mimic joint is set explicitly to
     ``ratio * driver`` (read from the URDF). Skipping this silently corrupts every
     distal link's FK. See :meth:`L20Kinematics.set_config`.
  2. **ADR-0003 segment directions.** Robot proximal/distal unit bones per finger
     from URDF link-frame origins (``worldLinkFramePosition``), in the hand-base
     frame (base is fixed at the world origin).

FK here is validated to match the yourdfpy FK that the solver/oracle were built on
to ~1e-8 (see tests/g1_kinematic).
"""
from __future__ import annotations

import os
import xml.etree.ElementTree as ET

import numpy as np
import pybullet as pb

from src.kinematics import L20FK

from .conventions import ACTIVE_IDX, JOINT_NAME, BASE_LINK, N_JOINTS

_THIS = os.path.dirname(os.path.abspath(__file__))
_URDF_ROOT = os.path.join(_THIS, "urdf", "l20")


def urdf_path(side):
    return os.path.join(_URDF_ROOT, side, f"linkerhand_l20_{side}.urdf")


def _parse_mimics(path):
    """Return {mimic_joint_name: (driver_joint_name, multiplier, offset)}."""
    root = ET.parse(path).getroot()
    out = {}
    for j in root.findall("joint"):
        m = j.find("mimic")
        if m is not None:
            out[j.get("name")] = (
                m.get("joint"),
                float(m.get("multiplier", "1")),
                float(m.get("offset", "0")),
            )
    return out


class L20Kinematics:
    """Kinematic L20 model for one side ('right'/'left').

    Owns a private PyBullet DIRECT client. Call :meth:`close` (or use as a context
    manager) to release it.
    """

    def __init__(self, side="right"):
        if side not in BASE_LINK:
            raise ValueError(f"side must be 'right'/'left', got {side!r}")
        self.side = side
        self.path = urdf_path(side)
        # Metric/measurement FK comes from the shared authority (src/kinematics,
        # pure yourdfpy) so measured r_prox/r_dist match the solver/oracle exactly
        # (ADR-0005). PyBullet below stays for link FK, mimic checks, and (future)
        # dynamics/contact only -- it is no longer an FK authority.
        self._fk = L20FK(side)
        self._last_q = [0.0] * N_JOINTS
        self.cid = pb.connect(pb.DIRECT)
        self.body = pb.loadURDF(self.path, useFixedBase=True, physicsClientId=self.cid)

        # name <-> pybullet index maps (joint index == child-link index)
        self.jidx, self.lidx = {}, {}
        self._limit = {}  # joint name -> (lower, upper) from URDF
        for i in range(pb.getNumJoints(self.body, physicsClientId=self.cid)):
            info = pb.getJointInfo(self.body, i, physicsClientId=self.cid)
            jname = info[1].decode()
            child = info[12].decode()
            self.jidx[jname] = i
            self.lidx[child] = i
            self._limit[jname] = (float(info[8]), float(info[9]))

        # mimic map (PyBullet won't apply these — we do, manually)
        self.mimics = _parse_mimics(self.path)

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
        """{semantic_idx: (lo, hi)} for the 16 active DoF, read from the URDF."""
        return {idx: self._limit[JOINT_NAME[idx]] for idx in ACTIVE_IDX}

    def mimic_limits(self):
        """{mimic_joint_name: (lo, hi)} from the URDF (dependent-joint ranges)."""
        return {m: self._limit[m] for m in self.mimics}

    # -- configuration ---------------------------------------------------- #
    def set_config(self, joint_rad):
        """Place the hand at an l20_targets ``joint_rad`` (20-vector, radians).

        Sets the 16 active joints, then ENFORCES every URDF mimic joint manually as
        ``multiplier * driver + offset``. Returns the full {joint_name: angle} dict
        actually applied (drivers + mimics) for inspection/validation.
        """
        if len(joint_rad) != 20:
            raise ValueError(f"joint_rad must have 20 entries, got {len(joint_rad)}")

        self._last_q = [float(v) for v in joint_rad]
        applied = {}
        for idx in ACTIVE_IDX:
            jname = JOINT_NAME[idx]
            val = float(joint_rad[idx])
            pb.resetJointState(self.body, self.jidx[jname], val, physicsClientId=self.cid)
            applied[jname] = val

        # MANUAL MIMIC ENFORCEMENT (PyBullet ignores <mimic>).
        for mname, (driver, mult, off) in self.mimics.items():
            mval = mult * applied[driver] + off
            pb.resetJointState(self.body, self.jidx[mname], mval, physicsClientId=self.cid)
            applied[mname] = mval
        return applied

    def joint_angles(self):
        """Current {joint_name: angle} for ALL URDF joints (drivers + mimics)."""
        return {
            n: pb.getJointState(self.body, i, physicsClientId=self.cid)[0]
            for n, i in self.jidx.items()
        }

    # -- forward kinematics ----------------------------------------------- #
    def link_origin(self, link_name):
        """URDF link-frame origin in the hand-base frame (3-vector)."""
        ls = pb.getLinkState(self.body, self.lidx[link_name],
                             computeForwardKinematics=True, physicsClientId=self.cid)
        return np.asarray(ls[4], dtype=float)  # worldLinkFramePosition

    def link_frame(self, link_name):
        """(origin(3), R(3x3)) of a URDF link frame in the hand-base frame.

        Unlike :meth:`link_origin`, the *orientation* R exposes the effect of a
        leaf joint (e.g. a coupled distal/``dip`` mimic) whose rotation does not
        move any downstream joint origin — useful to prove mimic enforcement is not
        a silent no-op.
        """
        ls = pb.getLinkState(self.body, self.lidx[link_name],
                             computeForwardKinematics=True, physicsClientId=self.cid)
        pos = np.asarray(ls[4], dtype=float)
        R = np.asarray(pb.getMatrixFromQuaternion(ls[5]), dtype=float).reshape(3, 3)
        return pos, R

    def segment_dirs(self):
        """{finger: (r_prox, r_dist)} unit bones at the current config (ADR-0003
        Finding-1: r_dist runs to the physical fingertip). Measured by the shared
        FK authority (src/kinematics) at the last-set config, so it matches the
        solver/oracle exactly. PyBullet FK is verified to agree (test_fk_authority),
        but the fingertip endpoint needs the baked tip offset, which lives in the
        authority -- not in raw PyBullet link origins."""
        return self._fk.all_segment_dirs(self._last_q)
