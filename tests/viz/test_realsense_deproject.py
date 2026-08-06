"""RealSense backend: deprojection + depth-hole handling + contract conformance.

All camera-free: mocked intrinsics + a synthetic depth map exercise the pure
deproject/sample helpers, and a fake detection drives the assembly + the existing
perception pipeline to a schema-valid hand_landmarks dict with confidence HIGH.
pyrealsense2 / mediapipe / opencv are never imported (the heavy deps live behind
the constructor, which these tests never call).
"""
import json
import os

import numpy as np
import pytest

from src.perception.realsense_source import (
    CameraIntrinsics, HIGH_DEPTH_CONFIDENCE, deproject_pixel, sample_depth_m,
    deproject_landmarks,
)
from src.perception.indices import N_LANDMARKS

HERE = os.path.dirname(os.path.abspath(__file__))
CONTRACT = os.path.normpath(os.path.join(
    HERE, os.pardir, os.pardir, "contracts", "hand_landmarks.schema.json"))

# A plausible 640x480 colour intrinsic (principal point at centre).
INTR = CameraIntrinsics(width=640, height=480, fx=600.0, fy=600.0,
                        ppx=320.0, ppy=240.0)
DEPTH_SCALE = 0.001  # z16 millimetres -> metres


# --- deprojection math ----------------------------------------------------- #
def test_deproject_principal_point_is_on_axis():
    """A pixel at the principal point deprojects to (0, 0, depth)."""
    p = deproject_pixel(INTR, INTR.ppx, INTR.ppy, 0.5)
    assert np.allclose(p, [0.0, 0.0, 0.5])


def test_deproject_known_offset():
    """Hand-computed pinhole back-projection for an off-centre pixel."""
    px, py, z = 380.0, 300.0, 0.8
    p = deproject_pixel(INTR, px, py, z)
    expect = np.array([z * (px - 320.0) / 600.0,
                       z * (py - 240.0) / 600.0, z])
    assert np.allclose(p, expect)
    # +x to the right of centre, +y below centre, +z forward.
    assert p[0] > 0 and p[1] > 0 and p[2] == pytest.approx(0.8)


def test_deproject_roundtrip_against_projection():
    """Project a known 3D point with the pinhole model, then deproject back."""
    pt = np.array([0.05, -0.03, 0.6])
    px = INTR.ppx + INTR.fx * pt[0] / pt[2]
    py = INTR.ppy + INTR.fy * pt[1] / pt[2]
    back = deproject_pixel(INTR, px, py, pt[2])
    assert np.allclose(back, pt, atol=1e-9)


# --- depth sampling + hole handling ---------------------------------------- #
def _depth_map(fill_mm=700):
    return np.full((INTR.height, INTR.width), fill_mm, dtype=np.uint16)


def test_sample_direct_hit():
    d, ok = sample_depth_m(_depth_map(700), 320.0, 240.0, DEPTH_SCALE)
    assert ok and d == pytest.approx(0.700)


def test_sample_hole_uses_neighbourhood_median():
    dm = _depth_map(700)
    dm[240, 320] = 0  # a hole at the sampled pixel; neighbours are valid
    d, ok = sample_depth_m(dm, 320.0, 240.0, DEPTH_SCALE, win=2)
    assert ok and d == pytest.approx(0.700)


def test_sample_total_hole_unresolved_never_nan():
    dm = np.zeros((INTR.height, INTR.width), dtype=np.uint16)  # all holes
    d, ok = sample_depth_m(dm, 320.0, 240.0, DEPTH_SCALE)
    assert not ok
    assert np.isfinite(d)


def test_sample_out_of_bounds_pixel_is_clamped():
    dm = _depth_map(500)
    d, ok = sample_depth_m(dm, 99999.0, -50.0, DEPTH_SCALE)
    assert ok and d == pytest.approx(0.500) and np.isfinite(d)


# --- landmark deprojection + hole policy ----------------------------------- #
def _grid_landmarks():
    """21 landmark pixels spread across the image."""
    rng = np.random.default_rng(0)
    xs = rng.uniform(100, 540, size=N_LANDMARKS)
    ys = rng.uniform(80, 400, size=N_LANDMARKS)
    return list(zip(xs.tolist(), ys.tolist()))


def test_deproject_landmarks_all_valid():
    px = _grid_landmarks()
    dm = _depth_map(650)
    pts, n_holes, depths = deproject_landmarks(px, dm, INTR, DEPTH_SCALE)
    assert pts.shape == (N_LANDMARKS, 3)
    assert n_holes == 0
    assert np.all(np.isfinite(pts))
    assert np.allclose(pts[:, 2], 0.650)  # z == measured metric depth


def test_deproject_landmarks_holes_filled_finite_and_flagged():
    px = _grid_landmarks()
    dm = _depth_map(650)
    # punch isolated holes (no valid neighbour) at three landmark pixels
    for i in (0, 8, 20):
        xi, yi = int(round(px[i][0])), int(round(px[i][1]))
        dm[yi - 3:yi + 4, xi - 3:xi + 4] = 0
    pts, n_holes, depths = deproject_landmarks(px, dm, INTR, DEPTH_SCALE)
    assert n_holes == 3                      # flagged
    assert np.all(np.isfinite(pts))          # never NaN
    # holes borrowed the resolved-landmark median depth (0.650)
    assert depths[0] == pytest.approx(0.650)


def test_deproject_landmarks_hold_last_depth():
    px = _grid_landmarks()
    dm_all_holes = np.zeros((INTR.height, INTR.width), dtype=np.uint16)
    last = np.full(N_LANDMARKS, 0.55)
    pts, n_holes, depths = deproject_landmarks(
        px, dm_all_holes, INTR, DEPTH_SCALE, last_depths=last)
    assert n_holes == N_LANDMARKS
    assert np.all(np.isfinite(pts))
    assert np.allclose(depths, 0.55)         # reused the per-landmark history


def test_deproject_landmarks_all_holes_no_history_fallback_finite():
    from src.perception.realsense_source import _FALLBACK_DEPTH_M
    px = _grid_landmarks()
    dm = np.zeros((INTR.height, INTR.width), dtype=np.uint16)
    pts, n_holes, depths = deproject_landmarks(px, dm, INTR, DEPTH_SCALE)
    assert n_holes == N_LANDMARKS
    assert np.all(np.isfinite(pts))
    assert np.allclose(depths, _FALLBACK_DEPTH_M)


# --- contract conformance via the existing pipeline ------------------------ #
class _FakeRealSense:
    """Drives RealSenseHandSource._build_detection without touching hardware.

    We bind the unbound assembly method onto a minimal object carrying the few
    attributes it reads — no constructor (which would import pyrealsense2).
    """

    def __init__(self):
        self._depth_scale = DEPTH_SCALE
        self.hole_win = 2
        self.max_hole_frac = 0.5
        self.depth_confidence = HIGH_DEPTH_CONFIDENCE
        self._last_depths = None

    _build_detection = (
        __import__("src.perception.realsense_source", fromlist=["RealSenseHandSource"])
        .RealSenseHandSource._build_detection)


def _hand_pixels(side="right"):
    """Project a canonical flat hand (hand_base metres) into the image with a
    simple pinhole at a fixed standoff, giving realistic landmark pixels."""
    from src.perception.indices import CANONICAL_FLAT_RIGHT
    lm = CANONICAL_FLAT_RIGHT.copy()
    if side == "left":
        lm[:, 1] *= -1.0
    # place the hand 0.5 m in front, centred; camera +z forward
    cam = lm + np.array([0.0, 0.0, 0.5])
    px = [(INTR.ppx + INTR.fx * p[0] / p[2], INTR.ppy + INTR.fy * p[1] / p[2])
          for p in cam]
    depth_mm = np.array([p[2] for p in cam]) * 1000.0
    return px, depth_mm


def test_realsense_emits_valid_contract_high_confidence():
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.load(open(CONTRACT))

    px, depth_mm = _hand_pixels("right")
    # synthetic aligned depth map: write each landmark's metric depth at its pixel
    dm = np.zeros((INTR.height, INTR.width), dtype=np.uint16)
    for (x, y), z in zip(px, depth_mm):
        dm[int(round(y)), int(round(x))] = int(round(z))

    fake = _FakeRealSense()
    # MediaPipe labels camera-view; non-mirrored image -> swap (handedness.py).
    det = fake._build_detection(px, dm, INTR, t=0.0, handed_label="Left", score=0.97)
    assert det.ok
    assert det.depth_confidence == HIGH_DEPTH_CONFIDENCE
    assert det.landmarks.shape == (N_LANDMARKS, 3)
    assert np.all(np.isfinite(det.landmarks))

    # Feed through the EXISTING perception pipeline (untouched) -> contract dict.
    from src.perception.pipeline import HandPipeline
    from src.perception.source import ReplayHandSource
    pipe = HandPipeline(ReplayHandSource([det]), smoothing=False)
    out = list(pipe.run())
    assert len(out) == 1
    contract = out[0].to_contract()
    jsonschema.validate(contract, schema)        # schema-valid hand_landmarks
    assert contract["frame"] == "hand_base"
    assert contract["side"] == "right"           # swapped from camera-view "Left"
    assert contract["depth_confidence"] == HIGH_DEPTH_CONFIDENCE
    # HIGH confidence -> the pipeline does NOT raise the low-depth warning.
    assert "low_depth_confidence" not in contract["warnings"]


def test_realsense_excessive_holes_reports_not_ok():
    """Too many depth holes -> ok=False so the pipeline holds the last good frame
    (reusing the existing robustness path), rather than a mostly-filled cloud."""
    px = _grid_landmarks()
    dm = np.zeros((INTR.height, INTR.width), dtype=np.uint16)  # every landmark a hole
    fake = _FakeRealSense()
    det = fake._build_detection(px, dm, INTR, t=0.0, handed_label="Left", score=0.9)
    assert det.ok is False
