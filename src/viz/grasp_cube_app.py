"""Live MediaPipe -> dynamic LinkerHand grasp scene with a 5.5 cm cube.

This is the contact/dynamics version of ``src.viz.app`` for object interaction.
It keeps the same camera/retargeting path, but drives ``src.sim.dynamics`` with
position motors and a graspable cube instead of kinematically placing joints.
"""
from __future__ import annotations

import argparse
import time
from typing import Optional

import numpy as np

from src.sim.dynamics import L20Dynamics, PDGains
from src.sim.grasp import PALM_GRAVITY
from src.viz.app import (
    _camera_preview,
    _compose_adjust,
    _little_abd_adjuster,
    _thumb_adjuster,
    _thumb_grasp_adjuster,
    _thumb_only_adjuster,
    _thumb_orient_adjuster,
)
from src.viz.core import DEFAULT_DT, teleop_command


CUBE_SIZE_M = 0.055


def run_webcam_cube_grasp(
    camera_index: int = 0,
    side: Optional[str] = "right",
    *,
    fps: float = 30.0,
    show_camera: bool = False,
    use_filter: bool = True,
    image_mirrored: bool = False,
    smoothing: bool = True,
    one_euro_min_cutoff: float = 1.5,
    one_euro_beta: float = 0.05,
    one_euro_d_cutoff: float = 1.0,
    cube_size: float = CUBE_SIZE_M,
    cube_pos=(0.065, 0.0, 0.115),
    cube_mass: float = 0.035,
    cube_friction: float = 1.6,
    sim_hz: float = 240.0,
    sim_steps_per_frame: int = 8,
    max_frames: Optional[int] = None,
    thumb_gain: float = 1.0,
    thumb_cross_gain: float = 0.0,
    little_abd_gain: float = 1.0,
    thumb_grasp_gain: float = 0.25,
    thumb_orient_gain: float = 0.6,
    thumb_only: bool = False,
    kp: float = 0.35,
    kd: float = 0.7,
    max_force_nm: float = 0.16,
) -> None:
    from src.perception.mediapipe_source import MediaPipeHandSource
    from src.perception.pipeline import HandPipeline
    from src.perception.one_euro import OneEuroConfig

    source = MediaPipeHandSource(camera_index=camera_index, fps=fps)
    pipeline = HandPipeline(
        source,
        smoothing=smoothing,
        one_euro=OneEuroConfig(
            min_cutoff=one_euro_min_cutoff,
            beta=one_euro_beta,
            d_cutoff=one_euro_d_cutoff,
        ),
        image_mirrored=image_mirrored,
        force_side=side,
    )
    gains = PDGains(kp=kp, kd=kd, max_force_nm=max_force_nm,
                    mimic_max_force_nm=max_force_nm * 1.4)
    dyn = L20Dynamics(side or "right", timestep=1.0 / sim_hz, gains=gains,
                      gravity=PALM_GRAVITY, gui=True)
    cube_half = float(cube_size) / 2.0
    dyn.add_box("cube_55mm", [cube_half, cube_half, cube_half], list(cube_pos),
                mass=cube_mass, lateral_friction=cube_friction)

    adjust = _compose_adjust(
        _thumb_adjuster(thumb_gain, thumb_cross_gain),
        _little_abd_adjuster(little_abd_gain),
        _thumb_grasp_adjuster(thumb_grasp_gain),
        _thumb_orient_adjuster(thumb_orient_gain),
        _thumb_only_adjuster(thumb_only),
    )

    cv2 = None
    if show_camera:
        import cv2  # noqa: F401

    prev_safe = None
    last_side = None
    frames = 0
    try:
        for det in source:
            pf = pipeline.process(det)
            if pf is None:
                if not _camera_preview(cv2, source, "Webcam camera", None):
                    break
                dyn.step(sim_steps_per_frame)
                time.sleep(DEFAULT_DT)
                continue

            if pf.side != last_side:
                prev_safe = None
                last_side = pf.side
            if not _camera_preview(cv2, source, "Webcam camera", pf):
                break

            out = teleop_command(
                pf.landmarks,
                pf.side,
                prev_safe,
                DEFAULT_DT,
                use_filter=use_filter,
                candidate_adjust=adjust,
            )
            if out["safe"] is not None:
                prev_safe = out["safe"]
            dyn.set_command(out["command"])
            dyn.step(sim_steps_per_frame)

            pos, _ = dyn.object_pose("cube_55mm")
            force = dyn.max_contact_force("cube_55mm")
            if frames % 15 == 0:
                print(
                    f"[cube] pos={np.round(pos, 4).tolist()} "
                    f"contact_force={force:.2f}N",
                    flush=True,
                )

            frames += 1
            if max_frames is not None and frames >= max_frames:
                break
            time.sleep(DEFAULT_DT)
    finally:
        source.close()
        dyn.close()
        if cv2 is not None:
            cv2.destroyAllWindows()


def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Live MediaPipe dynamic LinkerHand grasp with a 5.5 cm cube.")
    ap.add_argument("--camera-index", type=int, default=0)
    ap.add_argument("--side", choices=("right", "left"), default="right")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--show-camera", action="store_true")
    ap.add_argument("--image-mirrored", action="store_true")
    ap.add_argument("--no-filter", action="store_true")
    ap.add_argument("--no-smoothing", action="store_true",
                    help="disable perception-side one-euro landmark smoothing")
    ap.add_argument("--one-euro-min-cutoff", type=float, default=1.5,
                    help="perception One Euro min cutoff in Hz; lower is smoother but laggier")
    ap.add_argument("--one-euro-beta", type=float, default=0.05,
                    help="perception One Euro speed coefficient; higher reduces lag while moving")
    ap.add_argument("--one-euro-d-cutoff", type=float, default=1.0,
                    help="perception One Euro derivative cutoff in Hz")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--cube-size", type=float, default=CUBE_SIZE_M)
    ap.add_argument("--cube-x", type=float, default=0.065)
    ap.add_argument("--cube-y", type=float, default=0.0)
    ap.add_argument("--cube-z", type=float, default=0.115)
    ap.add_argument("--cube-mass", type=float, default=0.035)
    ap.add_argument("--cube-friction", type=float, default=1.6)
    ap.add_argument("--sim-steps-per-frame", type=int, default=8)
    ap.add_argument("--thumb-gain", type=float, default=1.0)
    ap.add_argument("--thumb-cross-gain", type=float, default=0.0)
    ap.add_argument("--little-abd-gain", type=float, default=1.0)
    ap.add_argument("--thumb-grasp-gain", type=float, default=0.25)
    ap.add_argument("--thumb-orient-gain", type=float, default=0.6)
    ap.add_argument("--thumb-only", action="store_true")
    ap.add_argument("--kp", type=float, default=0.35)
    ap.add_argument("--kd", type=float, default=0.7)
    ap.add_argument("--max-force-nm", type=float, default=0.16)
    args = ap.parse_args(argv)

    run_webcam_cube_grasp(
        camera_index=args.camera_index,
        side=args.side,
        fps=args.fps,
        show_camera=args.show_camera,
        use_filter=not args.no_filter,
        image_mirrored=args.image_mirrored,
        smoothing=not args.no_smoothing,
        one_euro_min_cutoff=args.one_euro_min_cutoff,
        one_euro_beta=args.one_euro_beta,
        one_euro_d_cutoff=args.one_euro_d_cutoff,
        cube_size=args.cube_size,
        cube_pos=(args.cube_x, args.cube_y, args.cube_z),
        cube_mass=args.cube_mass,
        cube_friction=args.cube_friction,
        sim_steps_per_frame=args.sim_steps_per_frame,
        max_frames=args.max_frames,
        thumb_gain=args.thumb_gain,
        thumb_cross_gain=args.thumb_cross_gain,
        little_abd_gain=args.little_abd_gain,
        thumb_grasp_gain=args.thumb_grasp_gain,
        thumb_orient_gain=args.thumb_orient_gain,
        thumb_only=args.thumb_only,
        kp=args.kp,
        kd=args.kd,
        max_force_nm=args.max_force_nm,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
