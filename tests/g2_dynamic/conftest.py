"""Shared fixtures + helpers for the G2 dynamic suite."""
import glob
import json
import os

import numpy as np
import pytest


def max_pen_depth(model, q, margin):
    """Max overlap depth (m) at config ``q``. depth = (ra+rb+margin) - dist, so a
    contact with depth <= margin is within the safety buffer but NOT actually
    overlapping; depth > margin is a real surface penetration."""
    ps = model.penetrations(np.asarray(q, float), margin)
    return max((p.depth for p in ps), default=0.0)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "out")
REAL_DIR = os.path.join(os.path.dirname(HERE), "g1_kinematic", "fixtures", "real")
BASELINE_PATH = os.path.join(HERE, "baseline.json")

SIDES = ("right", "left")

with open(BASELINE_PATH) as _f:
    BASELINE = json.load(_f)


def _real_path(side):
    """The recorded sequence for a side (perception convergence point)."""
    hits = sorted(glob.glob(os.path.join(REAL_DIR, f"*{side}*.json")))
    return hits[0] if hits else None


def load_frames(side):
    """Return [landmarks(21x3)] for the DETECTED frames of the side's real seq."""
    path = _real_path(side)
    if path is None:
        return None
    doc = json.load(open(path))
    seq = doc["frames"] if isinstance(doc, dict) and "frames" in doc else doc
    return [fr["landmarks"] for fr in seq if fr.get("detected", True) is not False]


@pytest.fixture(scope="session")
def baseline():
    return BASELINE


@pytest.fixture
def frames_right():
    fr = load_frames("right")
    if not fr:
        pytest.skip("recorded real sequence (right) not present")
    return fr


@pytest.fixture
def frames_left():
    fr = load_frames("left")
    if not fr:
        pytest.skip("recorded real sequence (left) not present")
    return fr


def ensure_out():
    os.makedirs(OUT_DIR, exist_ok=True)
    return OUT_DIR
