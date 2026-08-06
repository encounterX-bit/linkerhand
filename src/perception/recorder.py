"""Record ``hand_landmarks`` sequences for offline replay.

This is the convergence point with ``sim-agent``: its G1 real-set residual test
replays the files written here. Each file is a JSON object with a ``frames`` list
where every element is a full ``hand_landmarks`` contract dict (so each frame
validates independently). Output lands in tests/g1_kinematic/fixtures/real/.
"""
from __future__ import annotations

import json
import os
from typing import List, Optional

import numpy as np

from .pipeline import ProcessedFrame

# repo_root/tests/g1_kinematic/fixtures/real
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.normpath(os.path.join(_HERE, "..", ".."))
DEFAULT_REAL_DIR = os.path.join(
    _REPO_ROOT, "tests", "g1_kinematic", "fixtures", "real"
)


class Recorder:
    def __init__(self, out_dir: str = DEFAULT_REAL_DIR):
        self.out_dir = out_dir
        self._frames: List[dict] = []

    def __len__(self) -> int:
        return len(self._frames)

    def add(self, pf: Optional[ProcessedFrame]) -> None:
        if pf is None:
            return
        self._frames.append(pf.to_contract())

    def add_contract(self, frame: dict) -> None:
        self._frames.append(frame)

    def save(self, name: str, source: str = "unknown") -> str:
        """Write the recorded sequence to ``<out_dir>/<name>.json``; returns path."""
        if not self._frames:
            raise ValueError("nothing recorded")
        sides = {f["side"] for f in self._frames}
        # sanity: every frame is contract-shaped and finite
        for f in self._frames:
            lm = np.asarray(f["landmarks"], dtype=float)
            if lm.shape != (21, 3) or not np.all(np.isfinite(lm)):
                raise ValueError("recorded a non-(21,3)/non-finite frame")
            if f.get("frame") != "hand_base":
                raise ValueError("recorded frame is not in hand_base")
        os.makedirs(self.out_dir, exist_ok=True)
        if not name.endswith(".json"):
            name += ".json"
        path = os.path.join(self.out_dir, name)
        obj = {
            "schema": "hand_landmarks",
            "frame": "hand_base",
            "side": sides.pop() if len(sides) == 1 else "mixed",
            "source": source,
            "n_frames": len(self._frames),
            "frames": self._frames,
        }
        with open(path, "w") as fh:
            json.dump(obj, fh, indent=2)
        return path

    @staticmethod
    def load(path: str) -> dict:
        with open(path) as fh:
            return json.load(fh)
