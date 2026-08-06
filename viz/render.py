"""PyBullet renderer for the L20 mirror — kinematic joint set + mimic enforcement.

Loads the vendored L20 URDF in a PyBullet **GUI** (or DIRECT, for headless tests)
client and places the 16 active joints with ``resetJointState`` (crispest for a
mirror — no PD lag, no contact; dynamics is Stage 2). PyBullet ignores URDF
``<mimic>`` tags, so the 5 coupled distal mimics are enforced manually using the
ratios read from the ``src/kinematics`` FK authority (``L20FK.mimics``) — the same
ratios the sim harness and FK use, never a second copy.

Read-only over ``src/kinematics`` (mimic ratios, joint/index maps) and ``src/sim``
(URDF path). This module owns only the GUI client + joint placement.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pybullet as pb

from src.kinematics import L20FK, ACTIVE_IDX, RESERVED_IDX, JOINT_NAME, N_JOINTS
from src.sim import urdf_path

# semantic index <-> URDF driver-joint name (driver joints are the 16 active DoF)
_NAME_TO_IDX: Dict[str, int] = {name: idx for idx, name in JOINT_NAME.items()}


class L20VizModel:
    """A loaded L20 in a PyBullet client, driven kinematically by a 20-vector.

    Parameters
    ----------
    side : 'right' | 'left'
    gui  : True -> ``p.GUI`` window (live mirror); False -> ``p.DIRECT`` (headless
           tests). Both apply identical joint placement + mimic enforcement.
    """

    def __init__(self, side: str = "right", *, gui: bool = True):
        self.side = side
        # FK authority — used ONLY for the mimic ratios (no FK eval here).
        self._fk = L20FK(side)
        self.mimics = self._fk.mimics            # {mimic_joint: (driver, mult, off)}

        self.cid = pb.connect(pb.GUI if gui else pb.DIRECT)
        if gui:
            # A clean mirror view: hide the debug panels, frame the hand.
            pb.configureDebugVisualizer(pb.COV_ENABLE_GUI, 0,
                                        physicsClientId=self.cid)
            pb.resetDebugVisualizerCamera(
                cameraDistance=0.35, cameraYaw=50, cameraPitch=-25,
                cameraTargetPosition=[0.0, 0.0, 0.08], physicsClientId=self.cid)
        self.body = pb.loadURDF(urdf_path(side), useFixedBase=True,
                                physicsClientId=self.cid)

        self.jidx: Dict[str, int] = {}
        for i in range(pb.getNumJoints(self.body, physicsClientId=self.cid)):
            info = pb.getJointInfo(self.body, i, physicsClientId=self.cid)
            self.jidx[info[1].decode()] = i

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

    # -- joint placement -------------------------------------------------- #
    def set_joints(self, joint_rad) -> None:
        """Place the hand at a 20-vector command (16 active + reserved=0) and
        enforce the 5 mimic joints as ``mult * driver + off`` (ratios from
        ``src/kinematics``). Reserved idx 11-14 are forced to 0."""
        q = np.asarray(joint_rad, dtype=float).reshape(-1)
        if q.shape[0] != N_JOINTS:
            raise ValueError(f"joint_rad must have {N_JOINTS} entries, got {q.shape[0]}")
        for idx in ACTIVE_IDX:
            jn = JOINT_NAME[idx]
            pb.resetJointState(self.body, self.jidx[jn], float(q[idx]),
                               physicsClientId=self.cid)
        for idx in RESERVED_IDX:
            jn = JOINT_NAME.get(idx)
            if jn in self.jidx:
                pb.resetJointState(self.body, self.jidx[jn], 0.0,
                                   physicsClientId=self.cid)
        # mimic enforcement (PyBullet ignores <mimic>): mult*driver + off.
        for mname, (driver, mult, off) in self.mimics.items():
            if mname not in self.jidx or driver not in _NAME_TO_IDX:
                continue
            dval = float(q[_NAME_TO_IDX[driver]])
            pb.resetJointState(self.body, self.jidx[mname], mult * dval + off,
                               physicsClientId=self.cid)

    def applied_active(self) -> List[float]:
        """Read back the 20-vector of active-joint angles (reserved/others 0).

        Lets a headless test confirm the kinematic set wired through correctly
        (resetJointState applied exactly the commanded active angles)."""
        out = [0.0] * N_JOINTS
        for idx in ACTIVE_IDX:
            jn = JOINT_NAME[idx]
            out[idx] = float(pb.getJointState(self.body, self.jidx[jn],
                                              physicsClientId=self.cid)[0])
        return out
