"""Lightweight, sim-free self-collision model for the L20 (capsules + palm slab).

Per the G2 ticket: build a *lightweight* collision proxy from the URDF collision
meshes — one capsule per phalanx plus a palm slab — and use the **src/kinematics
FK authority** for all link poses. No PyBullet, no sim dependency, no runtime
mesh loading (the geometry below is BAKED offline by ``_gen_collision_model.py``
from the vendored collision meshes; the URDF joints cannot express a capsule
radius or the palmar plane).

Geometry choices (see ADR-0008):
  * Phalanx capsule = the rigid phalanx LINK, a segment between two FK link
    origins (or origin->fingertip for the distal link) with a baked radius =
    half the smallest collision-mesh bounding-box extent (the cross-section).
  * Palm = a PALMAR HALF-PLANE (outward normal +x) with a y/z footprint, NOT a
    box. A box over the base-link AABB false-positives on a natural fist (the
    fingertips legitimately rest just outside the palmar skin); the half-plane
    only catches a tip driven *through* the palmar surface into the palm body,
    which on this hand is essentially a thumb-into-palm event (finger flexion is
    limit-bounded away from the palm).

The model exposes, per FK configuration, the list of penetrating constraints
with the Cartesian contact normal AND the analytic rigid-body Jacobian columns
(joint axis x lever) the filter needs to project the violation back into joint
space. The Jacobian is read off the FK link transforms — it is NOT a second FK.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.kinematics import L20FK, FINGERS, FINGER_ORDER, JOINT_NAME

# --- BAKED capsule radii (m), per side/link. Regenerate with
#     _gen_collision_model.py if the collision meshes change. ----------------- #
CAPSULE_RADII = {
    "right": {
        "thumb_metacarpals": 0.01545, "thumb_proximal": 0.00956, "thumb_distal": 0.00912,
        "index_proximal": 0.00849, "index_middle": 0.00819, "index_distal": 0.00770,
        "middle_proximal": 0.00849, "middle_middle": 0.00819, "middle_distal": 0.00770,
        "ring_proximal": 0.00849, "ring_middle": 0.00819, "ring_distal": 0.00770,
        "pinky_proximal": 0.00849, "pinky_middle": 0.00819, "pinky_distal": 0.00770,
    },
    "left": {
        "thumb_metacarpals": 0.01547, "thumb_proximal": 0.00956, "thumb_distal": 0.00912,
        "index_proximal": 0.00849, "index_middle": 0.00819, "index_distal": 0.00770,
        "middle_proximal": 0.00849, "middle_middle": 0.00819, "middle_distal": 0.00770,
        "ring_proximal": 0.00849, "ring_middle": 0.00819, "ring_distal": 0.00770,
        "pinky_proximal": 0.00849, "pinky_middle": 0.00819, "pinky_distal": 0.00770,
    },
}

# --- BAKED palm slab (palmar half-plane + y/z footprint), per side. The plane
#     normal points palmar-outward (+x); a tip is "inside the palm" when it lies
#     on the dorsal side AND within the footprint. x0 sits behind every natural
#     fist tip (measured min fist-tip x ~ 0.013) so a closed fist is collision-
#     free. y bounds mirror between sides (palm AABB). ------------------------ #
PALM_SLAB = {
    "right": {"normal": (1.0, 0.0, 0.0), "x0": -0.005,
              "y_lo": -0.0555, "y_hi": 0.0507, "z_lo": 0.0, "z_hi": 0.1689},
    "left":  {"normal": (1.0, 0.0, 0.0), "x0": -0.005,
              "y_lo": -0.0507, "y_hi": 0.0555, "z_lo": 0.0, "z_hi": 0.1689},
}


@dataclass(frozen=True)
class _Contrib:
    """One Jacobian contributor for a capsule's rigid link: a (possibly mimic-
    scaled) revolute joint that moves every point on that link."""

    active_idx: int      # the L20 active DoF this joint is driven by
    child_link: str      # joint origin/axis come from this link's FK frame
    axis_local: tuple    # joint axis in the joint frame (== child-link frame)
    ratio: float         # 1.0 for a driver, mimic multiplier for a dependent


@dataclass(frozen=True)
class _Capsule:
    """A phalanx capsule: a segment (two link names; hi may be the fingertip)
    with a baked radius and the joints that move it (rigid-body Jacobian)."""

    finger: str
    name: str            # "prox" | "mid" | "dist"
    lo_link: str
    hi_link: str | None  # None -> the segment's hi endpoint is the fingertip
    radius: float
    contribs: tuple      # tuple[_Contrib]
    is_distal: bool


@dataclass
class Penetration:
    """One violated non-penetration constraint at a configuration."""

    kind: str                  # "finger-finger" | "thumb-finger" | "tip-palm"
    normal: np.ndarray         # unit Cartesian normal, push direction for pa
    depth: float               # required_sep - dist  (> 0)
    # Jacobian gradient of the (signed) separation w.r.t. the 20 active joints:
    grad: np.ndarray           # shape (20,)


def _seg_seg_closest(p1, q1, p2, q2):
    """Closest points between segments [p1,q1] and [p2,q2]; returns (c1, c2)."""
    d1 = q1 - p1
    d2 = q2 - p2
    r = p1 - p2
    a = float(d1 @ d1)
    e = float(d2 @ d2)
    f = float(d2 @ r)
    if a < 1e-18 and e < 1e-18:
        return p1, p2
    if a < 1e-18:
        s = 0.0
        t = np.clip(f / e, 0.0, 1.0)
    else:
        c = float(d1 @ r)
        if e < 1e-18:
            t = 0.0
            s = np.clip(-c / a, 0.0, 1.0)
        else:
            b = float(d1 @ d2)
            den = a * e - b * b
            s = np.clip((b * f - c * e) / den, 0.0, 1.0) if den > 1e-18 else 0.0
            t = (b * s + f) / e
            if t < 0.0:
                t = 0.0
                s = np.clip(-c / a, 0.0, 1.0)
            elif t > 1.0:
                t = 1.0
                s = np.clip((b - c) / a, 0.0, 1.0)
    return p1 + d1 * s, p2 + d2 * t


class CollisionModel:
    """Capsule + palm model for one side, driven by the src/kinematics FK."""

    def __init__(self, side: str = "right"):
        self.side = side
        self.fk = L20FK(side)
        self._radii = CAPSULE_RADII[side]
        self._palm = PALM_SLAB[side]
        self._name_to_idx = {jn: idx for idx, jn in JOINT_NAME.items()}
        self._capsules = self._build_capsules()
        # adjacency for inter-finger checks (ticket: adjacent fingers only)
        self._adjacent = [("index", "middle"), ("middle", "ring"), ("ring", "little")]
        self._fingers = [f for f in FINGER_ORDER if f != "thumb"]

    # -- static build (once) ------------------------------------------------- #
    def _joint_child_map(self):
        jm = self.fk._urdf.joint_map
        return {jn: jm[jn].child for jn in jm}, jm

    def _contribs_for_link(self, rigid_link, jm) -> tuple:
        """Walk parent joints from ``rigid_link`` up to base; each actuated or
        mimic joint encountered moves this link rigidly. Mimic joints are
        attributed to their driver's active idx, scaled by the mimic ratio."""
        child_to_joint = {j.child: jn for jn, j in jm.items()}
        contribs = []
        link = rigid_link
        while link in child_to_joint:
            jn = child_to_joint[link]
            j = jm[jn]
            if jn in self._name_to_idx:                      # actuated driver
                contribs.append(_Contrib(self._name_to_idx[jn], j.child,
                                         tuple(float(x) for x in j.axis), 1.0))
            elif jn in self.fk.mimics:                        # dependent (mimic)
                driver, mult, _off = self.fk.mimics[jn]
                if driver in self._name_to_idx:
                    contribs.append(_Contrib(self._name_to_idx[driver], j.child,
                                             tuple(float(x) for x in j.axis), mult))
            link = j.parent
        return tuple(contribs)

    def _build_capsules(self) -> dict:
        _child, jm = self._joint_child_map()
        caps = {}
        for fn, spec in FINGERS.items():
            links = [spec.link_a, spec.link_b, spec.link_c]
            names = ["prox", "mid", "dist"]
            flist = []
            for i, (nm, lk) in enumerate(zip(names, links)):
                is_distal = (i == 2)
                hi = None if is_distal else links[i + 1]
                flist.append(_Capsule(
                    finger=fn, name=nm, lo_link=lk, hi_link=hi,
                    radius=self._radii[lk],
                    contribs=self._contribs_for_link(lk, jm),
                    is_distal=is_distal,
                ))
            caps[fn] = flist
        return caps

    # -- per-configuration geometry ----------------------------------------- #
    def _frames(self):
        """{link: (origin(3), R(3x3))} for every link, at the FK's current cfg."""
        out = {}
        for ln in self.fk._urdf.link_map:
            T = self.fk.transform(ln)
            out[ln] = (T[:3, 3], T[:3, :3])
        return out

    def _capsule_endpoints(self, cap, frames):
        lo = frames[cap.lo_link][0]
        if cap.hi_link is None:
            spec = FINGERS[cap.finger]
            hi = self.fk.fingertip(spec)        # uses current cfg
        else:
            hi = frames[cap.hi_link][0]
        return lo, hi

    def _jac_col(self, contribs, point, frames):
        """Rigid-body Jacobian of ``point`` (on a link) as a (20,3) array: for
        each contributor joint, ratio * (axis_world x (point - joint_origin))."""
        J = np.zeros((20, 3))
        for c in contribs:
            o, R = frames[c.child_link]
            axis_w = R @ np.asarray(c.axis_local)
            J[c.active_idx] += c.ratio * np.cross(axis_w, point - o)
        return J

    def penetrations(self, joint_rad, margin: float):
        """List[Penetration] at ``joint_rad`` with separation ``margin``.

        Each penetration carries the gradient of signed separation w.r.t. the 20
        active joints, so the filter can do a Jacobian-transpose projection.
        """
        self.fk.set_joint_rad(joint_rad)
        frames = self._frames()
        out = []

        def cap_cap(ca, cb, kind):
            a0, a1 = self._capsule_endpoints(ca, frames)
            b0, b1 = self._capsule_endpoints(cb, frames)
            pa, pb = _seg_seg_closest(a0, a1, b0, b1)
            delta = pa - pb
            dist = float(np.linalg.norm(delta))
            req = ca.radius + cb.radius + margin
            if dist >= req:
                return
            n = delta / dist if dist > 1e-9 else np.array([0.0, 0.0, 1.0])
            Ja = self._jac_col(ca.contribs, pa, frames)
            Jb = self._jac_col(cb.contribs, pb, frames)
            grad = Ja @ n - Jb @ n            # d(sep)/dq, separation = n.(pa-pb)
            out.append(Penetration(kind, n, req - dist, grad))

        if self.cfg_adjacent:
            for fa, fb in self._adjacent:
                for ca in self._capsules[fa]:
                    for cb in self._capsules[fb]:
                        cap_cap(ca, cb, "finger-finger")
        if self.cfg_thumb:
            for fb in self._fingers:
                for ca in self._capsules["thumb"]:
                    for cb in self._capsules[fb]:
                        cap_cap(ca, cb, "thumb-finger")
        if self.cfg_palm:
            n = np.asarray(self._palm["normal"], dtype=float)
            for fn in FINGER_ORDER:
                cap = self._capsules[fn][2]   # distal capsule
                _lo, tip = self._capsule_endpoints(cap, frames)
                # only active within the palm's y/z footprint
                if not (self._palm["y_lo"] <= tip[1] <= self._palm["y_hi"]
                        and self._palm["z_lo"] <= tip[2] <= self._palm["z_hi"]):
                    continue
                signed = float(n @ tip) - self._palm["x0"]   # +ve = palmar side
                req = cap.radius + margin
                if signed >= req:
                    continue
                J = self._jac_col(cap.contribs, tip, frames)
                grad = J @ n
                out.append(Penetration("tip-palm", n, req - signed, grad))
        return out

    # toggles set by the filter from its SafetyConfig (kept as attributes so the
    # model stays a pure function of (cfg, joint_rad)).
    cfg_adjacent = True
    cfg_thumb = True
    cfg_palm = True

    def configure(self, *, adjacent: bool, thumb: bool, palm: bool):
        self.cfg_adjacent = adjacent
        self.cfg_thumb = thumb
        self.cfg_palm = palm

    def max_penetration(self, joint_rad, margin: float = 0.0) -> float:
        """Deepest penetration depth at ``joint_rad`` (0.0 if collision-free).
        Used by tests to assert the OUTPUT is collision-free."""
        pens = self.penetrations(joint_rad, margin)
        return max((p.depth for p in pens), default=0.0)
