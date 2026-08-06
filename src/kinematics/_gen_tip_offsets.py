"""OFFLINE codegen: compute the fingertip local offsets baked in conventions.py.

NOT imported at runtime (FK must not load meshes). Run only to regenerate
``conventions.TIP_LOCAL`` when the vendored distal meshes change:

    python -m src.kinematics._gen_tip_offsets

For each side/finger, the fingertip is the distal-link mesh vertex farthest from
that link's own origin (the DIP/IP joint), expressed in the distal link's LOCAL
frame. This is the anatomical fingertip of a rounded distal pad; it is a body-
fixed point, so FK (which carries the DIP/IP mimic curl) moves it correctly.
"""
from __future__ import annotations

import os

import numpy as np
import trimesh
import yourdfpy

from .conventions import FINGERS, FINGER_ORDER, URDF_PATHS


def compute(side: str) -> dict:
    root = os.path.dirname(URDF_PATHS[side])
    u = yourdfpy.URDF.load(URDF_PATHS[side], load_meshes=True,
                           build_collision_scene_graph=False,
                           load_collision_meshes=False)
    out = {}
    for name in FINGER_ORDER:
        link = FINGERS[name].distal_link
        fn = u.link_map[link].visuals[0].geometry.mesh.filename
        mesh = trimesh.load(os.path.join(root, fn), force="mesh")
        v = np.asarray(mesh.vertices, dtype=float)
        tip = v[np.argmax(np.linalg.norm(v, axis=1))]
        out[name] = tuple(round(float(x), 10) for x in tip)
    return out


if __name__ == "__main__":
    print("TIP_LOCAL = {")
    for side in ("right", "left"):
        print(f"    {side!r}: {{")
        for name, off in compute(side).items():
            print(f"        {name!r}: {off!r},")
        print("    },")
    print("}")
