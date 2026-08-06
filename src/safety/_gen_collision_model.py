"""Offline generator for the baked collision constants in ``collision_model.py``.

Run this only when the vendored L20 collision meshes change. It loads the STL
collision meshes with ``trimesh`` (a dev/offline dependency — NOT imported at
runtime; the filter must stay mesh-free and sim-free) and prints:

  * ``CAPSULE_RADII`` — per phalanx link, radius = half the SMALLEST bounding-box
    extent (the cross-section perpendicular to the long bone axis). Half the
    smallest extent (rather than the mean of the two minor extents) keeps
    adjacent fingers — spaced ~0.022 m at rest — collision-free with a small
    separation margin, while still enclosing the bone.
  * ``PALM_SLAB`` — the base-link AABB, from which we take the palmar half-plane
    (normal +x) and the y/z footprint. x0 is then set BEHIND the measured
    minimum natural-fist fingertip x (~0.013 m) so a closed fist is collision-
    free (see ADR-0008); it is a hand-validated constant, not the raw AABB face.

Usage:  python -m safety._gen_collision_model   (from src/, with the venv active)
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import trimesh

# repo root on path so ``src.kinematics`` resolves (run from repo root):
#   python -m src.safety._gen_collision_model
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.kinematics import FINGERS, BASE_LINK  # noqa: E402

_URDF_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "sim", "urdf", "l20")


def _mesh_dir(side: str) -> str:
    return os.path.join(_URDF_DIR, side, "meshes")


def gen_side(side: str) -> dict:
    mdir = _mesh_dir(side)

    def load(link: str):
        return trimesh.load(os.path.join(mdir, link + ".STL"), force="mesh")

    radii = {}
    for spec in FINGERS.values():
        for link in (spec.link_a, spec.link_b, spec.link_c):
            ext = sorted(load(link).bounding_box.extents)
            radii[link] = round(float(ext[0]) / 2.0, 5)
    palm = load(BASE_LINK[side])
    bounds = palm.bounds  # [[xmin,ymin,zmin],[xmax,ymax,zmax]]
    return {"radii": radii, "palm_bounds": np.round(bounds, 4).tolist()}


if __name__ == "__main__":
    for side in ("right", "left"):
        r = gen_side(side)
        print(f"--- {side} ---")
        print("CAPSULE_RADII:", json.dumps(r["radii"]))
        print("palm AABB bounds:", r["palm_bounds"])
        print("PALM_SLAB normal=(1,0,0); x0 set behind min fist-tip x (~0.013) "
              "-> -0.005; footprint = palm y/z AABB.")
