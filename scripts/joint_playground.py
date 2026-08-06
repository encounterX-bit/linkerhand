#!/usr/bin/env python3
"""Drag sliders to pose the Linker Hand L20 in PyBullet. Sim-only, no pipeline.

This is a debug/inspection tool, NOT part of the teleop pipeline: no perception,
no retargeting, no safety filter. It just loads the vendored L20 URDF and gives
you one slider per actuated driver joint so you can see how the links move.

Driver joints, their limits, and the distal mimic couplings are all read from
the URDF at load time (the same source FK trusts) — nothing is hardcoded here,
so if the URDF changes the sliders follow. The 4 reserved DoF (idx 11-14) have
no URDF joint and simply never appear.

Usage:
    python scripts/joint_playground.py            # right hand (default)
    python scripts/joint_playground.py --side left
    python scripts/joint_playground.py --side right
"""
from __future__ import annotations

import argparse
import os
import time
import xml.etree.ElementTree as ET

import pybullet as p
import pybullet_data

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
URDF = {
    "right": os.path.join(REPO, "src/sim/urdf/l20/right/linkerhand_l20_right.urdf"),
    "left": os.path.join(REPO, "src/sim/urdf/l20/left/linkerhand_l20_left.urdf"),
}
# Unlimited/continuous joints get a sane slider range instead of +-inf.
DEFAULT_RANGE = (-1.57, 1.57)


def parse_mimics(urdf_path: str) -> dict[str, tuple[str, float, float]]:
    """{mimic_joint_name: (driver_joint_name, multiplier, offset)} from the URDF."""
    mimics = {}
    for j in ET.parse(urdf_path).getroot().findall("joint"):
        m = j.find("mimic")
        if m is not None:
            mimics[j.get("name")] = (
                m.get("joint"),
                float(m.get("multiplier", 1.0)),
                float(m.get("offset", 0.0)),
            )
    return mimics


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--side", choices=("right", "left"), default="right")
    args = ap.parse_args()
    urdf_path = URDF[args.side]

    mimics = parse_mimics(urdf_path)

    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
    p.resetDebugVisualizerCamera(
        cameraDistance=0.35, cameraYaw=50, cameraPitch=-30,
        cameraTargetPosition=[0, 0, 0.05],
    )
    hand = p.loadURDF(urdf_path, useFixedBase=True)

    # Map joint name -> pybullet index, and classify driver vs mimic.
    name_to_idx, driver_idx, sliders = {}, [], {}
    for j in range(p.getNumJoints(hand)):
        info = p.getJointInfo(hand, j)
        name = info[1].decode()
        jtype = info[2]
        name_to_idx[name] = j
        if jtype == p.JOINT_FIXED:
            continue
        if name in mimics:
            continue  # mimic links follow their driver; no slider
        lo, hi = info[8], info[9]
        if lo >= hi:  # unlimited joint -> sane default span
            lo, hi = DEFAULT_RANGE
        driver_idx.append(j)
        sliders[j] = (name, p.addUserDebugParameter(name, lo, hi, 0.0))

    # Pre-resolve mimic couplings into pybullet indices keyed by driver index.
    coupling: dict[int, list[tuple[int, float, float]]] = {}
    for mname, (driver, mult, off) in mimics.items():
        if mname in name_to_idx and driver in name_to_idx:
            coupling.setdefault(name_to_idx[driver], []).append(
                (name_to_idx[mname], mult, off)
            )

    # Disable the default velocity motors so resetJointState poses hold.
    for j in driver_idx + [c[0] for cs in coupling.values() for c in cs]:
        p.setJointMotorControl2(hand, j, p.VELOCITY_CONTROL, force=0)

    print(f"[joint_playground] {args.side} hand: {len(sliders)} driver sliders, "
          f"{len(mimics)} mimic joints coupled. Drag to pose; Ctrl-C to quit.")

    try:
        while True:
            for j, (name, s) in sliders.items():
                val = p.readUserDebugParameter(s)
                p.resetJointState(hand, j, val)
                for mj, mult, off in coupling.get(j, []):
                    p.resetJointState(hand, mj, mult * val + off)
            p.stepSimulation()
            time.sleep(1 / 120)
    except KeyboardInterrupt:
        print("\n[joint_playground] bye")
        p.disconnect()


if __name__ == "__main__":
    main()
