"""Sim-independent forward kinematics — the ONE FK authority for the L20.

This is the pure (yourdfpy-backed, analytic) forward kinematics shared by the
oracle (``eval/reference_solver``), the sim metric (``src/sim``), and the
solver's offline codegen (``src/finger_retarget/gen_constants``). PyBullet is NO
LONGER an FK authority — it stays in ``src/sim`` for dynamics/contact only (see
ADR-0005).

yourdfpy honours URDF ``<mimic>`` automatically, so the coupled DIP/IP curl is
applied for free here. Joint *limits* and *mimic ratios* are read from the URDF
at load (single source of truth); only the fingertip offset is a baked
convention constant (``conventions.TIP_LOCAL``), since the URDF joints cannot
express the tip of a leaf link.

Segment convention (ADR-0003, Finding-1 endpoint update):
    r_prox = unit(P_b - P_a)          # set by the BASE DoF
    r_dist = unit(fingertip - P_b)    # to the PHYSICAL FINGERTIP (mimic curl in)
where P_a, P_b are the link-A/link-B origins and ``fingertip`` is the distal
link frame applied to the baked local tip offset.
"""
from __future__ import annotations

import numpy as np
import yourdfpy

from .conventions import (
    ACTIVE_IDX, RESERVED_IDX, N_JOINTS, FINGER_ORDER,
    BASE_LINK, URDF_PATHS, FINGERS, FingerSpec, TIP_LOCAL,
)


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < 1e-12:
        return np.array([0.0, 0.0, 1.0])
    return np.asarray(v, dtype=float) / n


class L20FK:
    """Pure FK for one side ('right'/'left').

    Loads the vendored L20 URDF via yourdfpy (no meshes; the fingertip offset is
    baked). Exposes link transforms, per-finger segment directions, joint limits,
    and mimic ratios.
    """

    def __init__(self, side: str = "right"):
        if side not in URDF_PATHS:
            raise ValueError(f"side must be 'left'/'right', got {side!r}")
        self.side = side
        self.base_link = BASE_LINK[side]
        self._tip_local = {f: np.asarray(TIP_LOCAL[side][f], dtype=float)
                           for f in FINGER_ORDER}
        self._urdf = yourdfpy.URDF.load(
            URDF_PATHS[side],
            load_meshes=False,
            build_scene_graph=True,
            build_collision_scene_graph=False,
            load_collision_meshes=False,
        )
        self.actuated_joint_names = list(self._urdf.actuated_joint_names)
        # joint limits keyed by URDF joint name
        self.limits = {}
        for jn in self.actuated_joint_names:
            j = self._urdf.joint_map[jn]
            lo = j.limit.lower if j.limit is not None else -np.pi
            hi = j.limit.upper if j.limit is not None else np.pi
            self.limits[jn] = (float(lo), float(hi))
        # mimic map {mimic_joint: (driver, multiplier, offset)} from the URDF
        self.mimics = {}
        for jn, j in self._urdf.joint_map.items():
            m = getattr(j, "mimic", None)
            if m is not None:
                self.mimics[jn] = (m.joint, float(m.multiplier or 1.0),
                                   float(m.offset or 0.0))

    # -- configuration ---------------------------------------------------- #
    def set_cfg(self, joint_values: dict) -> None:
        """Set the configuration from a {joint_name: value} dict (others 0)."""
        cfg = {jn: 0.0 for jn in self.actuated_joint_names}
        for k, v in joint_values.items():
            if k in cfg:
                cfg[k] = float(v)
        self._urdf.update_cfg(cfg)

    def set_joint_rad(self, joint_rad) -> None:
        """Set the configuration from a 20-vector l20_targets ``joint_rad``."""
        if len(joint_rad) != N_JOINTS:
            raise ValueError(f"joint_rad must have {N_JOINTS} entries")
        jv = {}
        for spec in FINGERS.values():
            for idx, jn in spec.idx_to_joint().items():
                jv[jn] = float(joint_rad[idx])
        self.set_cfg(jv)

    # -- forward kinematics ----------------------------------------------- #
    def transform(self, link: str) -> np.ndarray:
        """4x4 homogeneous transform of ``link`` in the hand-base frame."""
        return self._urdf.get_transform(link, self.base_link)

    def link_origin(self, link: str) -> np.ndarray:
        return self.transform(link)[:3, 3]

    def fk(self, joint_rad) -> dict:
        """{link_name: 4x4 transform} for every link, at ``joint_rad``."""
        self.set_joint_rad(joint_rad)
        return {ln: self.transform(ln) for ln in self._urdf.link_map}

    def fingertip(self, spec: FingerSpec) -> np.ndarray:
        """Physical fingertip of ``spec`` in the hand-base frame (mimic curl in).

        The distal link transform already carries the DIP/IP mimic, so applying
        it to the baked local tip offset gives a fingertip that moves with the
        coupled distal joint.
        """
        T = self.transform(spec.distal_link)
        return T[:3, :3] @ self._tip_local[spec.name] + T[:3, 3]

    def segment_dirs(self, spec: FingerSpec, joint_values: dict | None = None):
        """(r_prox, r_dist) unit bones for ``spec`` (ADR-0003, fingertip r_dist).

        If ``joint_values`` is given the configuration is set first; otherwise the
        current configuration is used.
        """
        if joint_values is not None:
            self.set_cfg(joint_values)
        pa = self.link_origin(spec.link_a)
        pb = self.link_origin(spec.link_b)
        r_prox = _unit(pb - pa)
        r_dist = _unit(self.fingertip(spec) - pb)
        return r_prox, r_dist

    def segment_dirs_dip(self, spec: FingerSpec, joint_values: dict | None = None):
        """LEGACY pre-Finding-1 segment dirs: r_dist = unit(P_c - P_b) (the DIP
        origin bone). Retained only for the Step-1 extraction-faithfulness check
        against the historical oracle FK; NOT the canonical convention."""
        if joint_values is not None:
            self.set_cfg(joint_values)
        pa = self.link_origin(spec.link_a)
        pb = self.link_origin(spec.link_b)
        pc = self.link_origin(spec.link_c)
        return _unit(pb - pa), _unit(pc - pb)

    def all_segment_dirs(self, joint_rad) -> dict:
        """{finger: (r_prox, r_dist)} at a 20-vector ``joint_rad``."""
        self.set_joint_rad(joint_rad)
        return {name: self.segment_dirs(spec)
                for name, spec in FINGERS.items()}

    # -- limits ----------------------------------------------------------- #
    def finger_limits(self, spec: FingerSpec) -> dict:
        """{semantic_idx: (lo, hi)} for the actuated joints of this finger."""
        out = {
            spec.base_idx: self.limits[spec.base_joint],
            spec.abd_idx: self.limits[spec.abd_joint],
            spec.tip_idx: self.limits[spec.tip_joint],
        }
        if spec.opp_idx is not None:
            out[spec.opp_idx] = self.limits[spec.opp_joint]
        return out

    def active_limits(self) -> dict:
        """{semantic_idx: (lo, hi)} for all 16 active DoF."""
        out = {}
        for spec in FINGERS.values():
            out.update(self.finger_limits(spec))
        return out

    def mimic_limits(self) -> dict:
        """{mimic_joint_name: (lo, hi)} from the URDF (dependent-joint ranges)."""
        return {m: self.limits[m] for m in self.mimics if m in self.limits}
