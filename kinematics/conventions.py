"""Canonical L20 conventions — the ONE source for joint maps, segment maps,
landmark groups, mimic structure, and fingertip offsets.

Before this module existed the same convention was restated three times (the
oracle's ``FINGERS`` table, the sim harness's ``conventions.py``, and the
solver's ``gen_constants.py``). The kinematics refactor (ticket
``kinematics-agent-refactor``) collapses them here: oracle, sim, and the solver's
codegen all import this file, so the semantic mapping can never silently diverge.

What lives here (the parts the URDF does NOT encode):
  - the 16 actuated DoF layout (ACTIVE/RESERVED indices),
  - per-finger semantic-index -> URDF driver-joint mapping (``FINGERS``),
  - the ADR-0003 segment links (P_a, P_b, P_c) and human landmark groups,
  - the mesh-derived fingertip offset in each distal link's local frame
    (``TIP_LOCAL``) — needed for the Finding-1 fingertip-inclusive ``r_dist``.

Joint *limits* and *mimic ratios* are NOT hardcoded; they are read from the URDF
at load time (see ``fk.py``). This file fixes only naming/structure + the one
mesh-derived constant the URDF joints cannot express.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

# --- L20 actuated DoF layout (root CLAUDE.md / contracts/l20_targets) -------- #
ACTIVE_IDX = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 16, 17, 18, 19]
RESERVED_IDX = [11, 12, 13, 14]
N_JOINTS = 20
FINGER_ORDER = ["thumb", "index", "middle", "ring", "little"]

# Base link name differs by side (handedness; see hardware/LIMITS.md).
BASE_LINK = {"right": "base_link", "left": "hand_base_link"}

_URDF_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "src", "sim", "urdf", "l20",
)
URDF_PATHS = {
    "right": os.path.join(_URDF_DIR, "right", "linkerhand_l20_right.urdf"),
    "left": os.path.join(_URDF_DIR, "left", "linkerhand_l20_left.urdf"),
}


@dataclass(frozen=True)
class FingerSpec:
    """One finger's mapping from semantic L20 indices to URDF joints/links."""

    name: str
    landmarks: tuple  # (a, b, c, d) MediaPipe indices
    # semantic L20 joint indices this finger owns:
    base_idx: int
    abd_idx: int
    tip_idx: int
    opp_idx: int | None  # thumb only
    # URDF joint names driving each role (the 16 independent joints):
    base_joint: str
    abd_joint: str
    tip_joint: str
    opp_joint: str | None
    # URDF link names whose ORIGINS are P_a, P_b, P_c (link_c == distal link):
    link_a: str
    link_b: str
    link_c: str

    @property
    def is_thumb(self) -> bool:
        return self.opp_idx is not None

    @property
    def distal_link(self) -> str:
        """The last (fingertip-bearing) distal link — its tip is the fingertip."""
        return self.link_c

    @property
    def base_dof_joints(self) -> list:
        """Joints that orient r_prox (set the proximal direction)."""
        if self.is_thumb:
            # 3 CMC DoF orient the metacarpal (P_a -> P_b).
            return [self.opp_joint, self.abd_joint, self.base_joint]
        return [self.base_joint, self.abd_joint]

    @property
    def all_dof_joints(self) -> list:
        """All actuated joints for this finger, base DoF first, tip last."""
        return self.base_dof_joints + [self.tip_joint]

    @property
    def active_indices(self) -> list:
        idxs = [self.base_idx, self.abd_idx]
        if self.opp_idx is not None:
            idxs.append(self.opp_idx)
        idxs.append(self.tip_idx)
        return idxs

    def idx_to_joint(self) -> dict:
        """{semantic_idx: urdf_joint_name} for this finger's actuated DoF."""
        m = {
            self.base_idx: self.base_joint,
            self.abd_idx: self.abd_joint,
            self.tip_idx: self.tip_joint,
        }
        if self.opp_idx is not None:
            m[self.opp_idx] = self.opp_joint
        return m


# Joint names are identical between left/right URDFs EXCEPT the thumb distal,
# which is ``thumb_dip`` (right) / ``thumb_ip`` (left) -- but that is a mimic
# joint we never command, and the distal LINK is ``thumb_distal`` on both sides.
# P_a anchored at thumb_metacarpals (the cmc_pitch joint) so the thumb r_prox is
# a body-fixed vector in the metacarpals frame (the 3-CMC base solve is then an
# exact rotation alignment). See ADR-0003.
FINGERS: dict[str, FingerSpec] = {
    "thumb": FingerSpec(
        name="thumb", landmarks=(1, 2, 3, 4),
        base_idx=0, abd_idx=5, tip_idx=15, opp_idx=10,
        base_joint="thumb_cmc_pitch", abd_joint="thumb_cmc_roll",
        tip_joint="thumb_mcp", opp_joint="thumb_cmc_yaw",
        link_a="thumb_metacarpals", link_b="thumb_proximal",
        link_c="thumb_distal",
    ),
    "index": FingerSpec(
        name="index", landmarks=(5, 6, 7, 8),
        base_idx=1, abd_idx=6, tip_idx=16, opp_idx=None,
        base_joint="index_mcp_pitch", abd_joint="index_mcp_roll",
        tip_joint="index_pip", opp_joint=None,
        link_a="index_proximal", link_b="index_middle", link_c="index_distal",
    ),
    "middle": FingerSpec(
        name="middle", landmarks=(9, 10, 11, 12),
        base_idx=2, abd_idx=7, tip_idx=17, opp_idx=None,
        base_joint="middle_mcp_pitch", abd_joint="middle_mcp_roll",
        tip_joint="middle_pip", opp_joint=None,
        link_a="middle_proximal", link_b="middle_middle", link_c="middle_distal",
    ),
    "ring": FingerSpec(
        name="ring", landmarks=(13, 14, 15, 16),
        base_idx=3, abd_idx=8, tip_idx=18, opp_idx=None,
        base_joint="ring_mcp_pitch", abd_joint="ring_mcp_roll",
        tip_joint="ring_pip", opp_joint=None,
        link_a="ring_proximal", link_b="ring_middle", link_c="ring_distal",
    ),
    "little": FingerSpec(
        name="little", landmarks=(17, 18, 19, 20),
        base_idx=4, abd_idx=9, tip_idx=19, opp_idx=None,
        base_joint="pinky_mcp_pitch", abd_joint="pinky_mcp_roll",
        tip_joint="pinky_pip", opp_joint=None,
        link_a="pinky_proximal", link_b="pinky_middle", link_c="pinky_distal",
    ),
}

# --- derived convenience maps (so older call sites keep their shapes) -------- #
# semantic joint index -> independent (driver) URDF joint name.
JOINT_NAME = {idx: jn for spec in FINGERS.values()
              for idx, jn in spec.idx_to_joint().items()}

# Human side (ADR-0003): MediaPipe landmark group [a, b, c, d] per finger.
LANDMARK_GROUP = {name: spec.landmarks for name, spec in FINGERS.items()}

# Robot side (ADR-0003): three link-frame origins (P_a, P_b, P_c) per finger.
SEGMENT_LINKS = {name: (spec.link_a, spec.link_b, spec.link_c)
                 for name, spec in FINGERS.items()}

# --- fingertip offset (Finding-1) ------------------------------------------- #
# Per side/finger, the PHYSICAL FINGERTIP expressed in the distal link's LOCAL
# frame: the distal-link mesh vertex farthest from that link's own origin (the
# DIP/IP joint). FK transforms this offset by the distal link frame (which DOES
# carry the DIP/IP mimic curl), giving a fingertip that moves with the coupled
# distal joint — exactly what the Finding-1 ``r_dist`` needs. Baked here so FK
# never loads a mesh at runtime. Regenerate with ``_gen_tip_offsets.py`` if the
# URDF meshes change. (Mesh-derived; the URDF joints cannot express this point.)
TIP_LOCAL = {
    "right": {
        "thumb": (-0.0139459809, 0.0000000018, 0.0318848155),
        "index": (-0.0175404847, -0.0001473473, 0.0225316081),
        "middle": (-0.0175404828, -0.0001473553, 0.0225316118),
        "ring": (-0.0175404847, -0.0001473484, 0.0225316100),
        "little": (-0.0175404865, -0.0001473564, 0.0225316063),
    },
    "left": {
        "thumb": (-0.0151513182, -0.0010842372, 0.0313299187),
        "index": (-0.0175405182, 0.0000132070, 0.0225319881),
        "middle": (-0.0175405163, 0.0000132150, 0.0225319844),
        "ring": (-0.0175405145, 0.0000132081, 0.0225319825),
        "little": (-0.0175405163, 0.0000132161, 0.0225319862),
    },
}
