"""Phase-0 §3.1 + §3.2 + §3.3-structural discovery (CHEAP; no residual sweep).

Extracts, from the ACTUAL repo + URDF:
  - Linker Hand variant + DoF counts (independent vs mimic) straight from the URDF.
  - The 16 active joint rotation axes in the common hand-base frame at neutral pose.
  - Consecutive-axis angle classification per finger (perp / parallel / oblique).
  - The actuation/coupling (mimic) map with ratios.
  - DoF-vs-target determinacy per finger.
  - Solver timing instrumentation (per-call, per-finger) — the §3.2 evidence that
    the distal/thumb path is iterative, not bounded closed form.

Writes analysis/phase0/_discovery.json (consumed by the report writer).
Run: .venv/bin/python analysis/phase0/discover.py
"""
from __future__ import annotations

import json
import os
import sys
import time
import xml.etree.ElementTree as ET

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

from src.kinematics.conventions import (  # noqa: E402
    FINGERS, FINGER_ORDER, ACTIVE_IDX, RESERVED_IDX, URDF_PATHS, BASE_LINK,
)
from src.kinematics.fk import L20FK  # noqa: E402
from src.finger_retarget import retarget  # noqa: E402

TOL_DEG = 5.0  # ticket §3.3 classification tolerance


def classify(angle_deg):
    a = angle_deg
    if a <= TOL_DEG or a >= 180.0 - TOL_DEG:
        return "parallel"
    if abs(a - 90.0) <= TOL_DEG:
        return "perpendicular"
    return "oblique"


def parse_urdf_axes(path):
    """Return {joint_name: (type, axis_local, mimic|None, limit)} from raw URDF."""
    tree = ET.parse(path)
    root = tree.getroot()
    out = {}
    for j in root.findall("joint"):
        name = j.get("name")
        jtype = j.get("type")
        ax_el = j.find("axis")
        axis = tuple(float(x) for x in ax_el.get("xyz").split()) if ax_el is not None else None
        lim_el = j.find("limit")
        limit = None
        if lim_el is not None:
            limit = (float(lim_el.get("lower", "nan")), float(lim_el.get("upper", "nan")))
        m_el = j.find("mimic")
        mimic = None
        if m_el is not None:
            mimic = {
                "joint": m_el.get("joint"),
                "multiplier": float(m_el.get("multiplier", "1")),
                "offset": float(m_el.get("offset", "0")),
            }
        out[name] = {"type": jtype, "axis_local": axis, "mimic": mimic, "limit": limit}
    return out


def world_axis(fk: L20FK, joint_name, urdf_axes):
    """Rotation axis of a joint expressed in the hand-base frame at neutral pose."""
    # axis is defined in the joint's CHILD link frame; transform by that frame's
    # rotation in the base frame. yourdfpy joint child link == joint_map child.
    j = fk._urdf.joint_map[joint_name]
    child = j.child
    T = fk.transform(child)
    ax_local = np.asarray(urdf_axes[joint_name]["axis_local"], float)
    w = T[:3, :3] @ ax_local
    n = np.linalg.norm(w)
    return (w / n if n > 1e-12 else w)


def main():
    result = {"tol_deg": TOL_DEG, "sides": {}}
    # --- URDF-level facts (variant + DoF counts) from the RIGHT model ---------
    urdf_axes_r = parse_urdf_axes(URDF_PATHS["right"])
    revolute = [n for n, d in urdf_axes_r.items() if d["type"] == "revolute"]
    mimic = {n: d["mimic"] for n, d in urdf_axes_r.items() if d["mimic"]}
    independent = [n for n in revolute if not urdf_axes_r[n]["mimic"]]
    result["variant"] = {
        "model": "L20",
        "urdf_right": os.path.relpath(URDF_PATHS["right"], REPO),
        "n_revolute_joints": len(revolute),
        "n_mimic_joints": len(mimic),
        "n_independent_joints": len(independent),
        "n_active_dof": len(ACTIVE_IDX),
        "n_reserved_dof": len(RESERVED_IDX),
        "n_fingers": len(FINGER_ORDER),
    }
    result["mimic_map"] = {
        n: {"driver": m["joint"], "ratio": m["multiplier"], "offset": m["offset"]}
        for n, m in mimic.items()
    }

    for side in ("right", "left"):
        fk = L20FK(side)
        urdf_axes = parse_urdf_axes(URDF_PATHS[side])
        fk.set_joint_rad([0.0] * 20)  # neutral pose
        side_rec = {"fingers": {}}
        for name in FINGER_ORDER:
            spec = FINGERS[name]
            # ordered actuated chain proximal->distal:
            chain = spec.base_dof_joints + [spec.tip_joint]
            axes = {jn: world_axis(fk, jn, urdf_axes).tolist() for jn in chain}
            pairs = []
            for i in range(len(chain) - 1):
                a = np.asarray(axes[chain[i]])
                b = np.asarray(axes[chain[i + 1]])
                cos = float(np.clip(np.dot(a, b), -1, 1))
                ang = float(np.degrees(np.arccos(abs(cos))))  # axis is a line: fold to [0,90]
                ang_signed = float(np.degrees(np.arccos(cos)))
                pairs.append({
                    "from": chain[i], "to": chain[i + 1],
                    "angle_deg": round(ang_signed, 3),
                    "acute_angle_deg": round(ang, 3),
                    "class": classify(ang_signed),
                })
            # coupling: tip is driver + mimic
            tip_mimic = None
            for mn, m in mimic.items():
                if m["joint"] == spec.tip_joint:
                    tip_mimic = {"mimic_joint": mn, "ratio": m["multiplier"]}
            # DoF vs target
            n_base = len(spec.base_dof_joints)  # orients r_prox (2 DoF needed)
            side_rec["fingers"][name] = {
                "is_thumb": spec.is_thumb,
                "chain": chain,
                "axes_world": axes,
                "consecutive_pairs": pairs,
                "n_actuated_dof": len(chain),
                "n_base_dof": n_base,
                "tip_is_coupled": tip_mimic is not None,
                "tip_coupling": tip_mimic,
                "target_dim": 4,  # r_prox(2 free on sphere) + r_dist(2 free on sphere)
            }
        result["sides"][side] = side_rec

    # --- §3.2 timing instrumentation -----------------------------------------
    # Use the committed G0 fixtures + a small jittered batch (cheap; not the sweep).
    fix_dir = os.path.join(REPO, "tests", "g0_unit", "fixtures")
    poses = ["flat", "fist", "pinch", "point", "thumbs_up"]
    timing = {"per_call_us": {}, "n_calls": 0}
    rng = np.random.default_rng(0)
    all_t = []
    for side in ("right", "left"):
        for pose in poses:
            with open(os.path.join(fix_dir, f"{pose}_{side}.json")) as fh:
                lm0 = np.asarray(json.load(fh)["landmarks"], float)
            for k in range(50):  # 50 jittered variants per pose/side = 500 calls
                lm = lm0 + (rng.standard_normal(lm0.shape) * 0.002 if k else 0.0)
                t0 = time.perf_counter()
                retarget(lm, side=side)
                all_t.append((time.perf_counter() - t0) * 1e6)
    all_t = np.asarray(all_t)
    timing["n_calls"] = int(all_t.size)
    timing["per_call_us"] = {
        "p50": round(float(np.percentile(all_t, 50)), 1),
        "p95": round(float(np.percentile(all_t, 95)), 1),
        "p99": round(float(np.percentile(all_t, 99)), 1),
        "max": round(float(all_t.max()), 1),
        "mean": round(float(all_t.mean()), 1),
    }
    result["timing_fullhand"] = timing

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_discovery.json")
    with open(out, "w") as fh:
        json.dump(result, fh, indent=2)
    print("wrote", os.path.relpath(out, REPO))
    # console summary
    v = result["variant"]
    print(f"\nVARIANT: {v['model']}  revolute={v['n_revolute_joints']} "
          f"mimic={v['n_mimic_joints']} independent={v['n_independent_joints']} "
          f"active={v['n_active_dof']} reserved={v['n_reserved_dof']}")
    print("MIMIC MAP:")
    for n, m in result["mimic_map"].items():
        print(f"  {n} = {m['ratio']}*{m['driver']} + {m['offset']}")
    print("\nCONSECUTIVE-AXIS CLASSES (right):")
    for name, fr in result["sides"]["right"]["fingers"].items():
        cls = [f"{p['from'].split('_')[-1]}->{p['to'].split('_')[-1]}:{p['acute_angle_deg']:.1f}({p['class']})"
               for p in fr["consecutive_pairs"]]
        print(f"  {name:7s} {' '.join(cls)}")
    print("\nTIMING full-hand retarget() us:", result["timing_fullhand"]["per_call_us"])


if __name__ == "__main__":
    main()
