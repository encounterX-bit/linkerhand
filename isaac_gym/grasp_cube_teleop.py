"""Live MediaPipe -> Isaac Gym LinkerHand grasp demo with a 5.5 cm cube.

This script is intentionally standalone from the PyBullet visualizer: it reuses
the existing camera/perception/retargeting path, then writes the resulting
``joint_rad[20]`` command into Isaac Gym DOF position targets.

Run from the repo root, after activating an environment that has Isaac Gym:

    python -m src.isaac_gym.grasp_cube_teleop --source webcam --show-camera

The LinkerHand URDF contains mimic joints. Isaac Gym does not reliably enforce
URDF mimic tags at runtime, so this script expands the 20-vector command into
all Isaac Gym DOFs by name and applies each mimic target manually.
"""
from __future__ import annotations

import argparse
import os
import time
from typing import Dict, Iterable, Optional, Tuple

import numpy as np

from src.kinematics.conventions import ACTIVE_IDX, JOINT_NAME, N_JOINTS
from src.sim.kinematics import _parse_mimics, urdf_path
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


def _load_isaac_gym():
    try:
        from isaacgym import gymapi, gymutil  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "Isaac Gym is not installed in this Python environment. "
            "Activate a Python 3.8 Isaac Gym env first, or run with "
            "PYTHONPATH=/home/zhaoyan-qian/Desktop/Jacky/isaacgym/python:$PYTHONPATH. "
            "Isaac Gym Preview does not support Python 3.12."
        ) from exc
    return gymapi, gymutil


def _make_stream(args) -> Tuple[object, object, Iterable]:
    from src.perception.one_euro import OneEuroConfig

    if args.source == "webcam":
        from src.perception.mediapipe_source import MediaPipeHandSource
        from src.perception.pipeline import HandPipeline

        source = MediaPipeHandSource(camera_index=args.camera_index, fps=args.fps)
    elif args.source == "video":
        if not args.video_path:
            raise ValueError("--source video requires --video-path")
        from src.perception.video_source import VideoHandSource
        from src.perception.pipeline import HandPipeline

        source = VideoHandSource(args.video_path, fps=args.fps,
                                 playback_rate=args.playback_rate)
    else:
        from src.perception.realsense_source import RealSenseHandSource
        from src.perception.pipeline import HandPipeline

        source = RealSenseHandSource()

    pipeline = HandPipeline(
        source,
        smoothing=not args.no_smoothing,
        one_euro=OneEuroConfig(
            min_cutoff=args.one_euro_min_cutoff,
            beta=args.one_euro_beta,
            d_cutoff=args.one_euro_d_cutoff,
        ),
        image_mirrored=args.image_mirrored,
        force_side=args.side,
    )
    return source, pipeline, source


class IsaacGymLinkerHand:
    """Small Isaac Gym wrapper that accepts LinkerHand ``joint_rad[20]``."""

    def __init__(self, args):
        gymapi, _gymutil = _load_isaac_gym()
        self.gymapi = gymapi
        self.gym = gymapi.acquire_gym()
        self.args = args
        self.side = args.side
        self.mimics = _parse_mimics(urdf_path(self.side))
        self.driver_name_to_semantic = {name: idx for idx, name in JOINT_NAME.items()}

        sim_params = gymapi.SimParams()
        sim_params.dt = 1.0 / float(args.sim_hz)
        sim_params.substeps = int(args.substeps)
        sim_params.up_axis = gymapi.UP_AXIS_Z
        sim_params.gravity = gymapi.Vec3(args.gravity_x, args.gravity_y, args.gravity_z)
        sim_params.physx.solver_type = 1
        sim_params.physx.num_position_iterations = 8
        sim_params.physx.num_velocity_iterations = 2
        sim_params.physx.contact_offset = 0.004
        sim_params.physx.rest_offset = 0.0

        self.sim = self.gym.create_sim(
            int(args.compute_device_id),
            int(args.graphics_device_id),
            gymapi.SIM_PHYSX,
            sim_params,
        )
        if self.sim is None:
            raise RuntimeError("failed to create Isaac Gym sim")

        self.viewer = None
        if not args.headless:
            self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
            if self.viewer is None:
                raise RuntimeError("failed to create Isaac Gym viewer")

        self.env = self.gym.create_env(
            self.sim,
            gymapi.Vec3(-0.35, -0.35, -0.05),
            gymapi.Vec3(0.35, 0.35, 0.35),
            1,
        )
        self.hand_actor = self._create_hand()
        self.cube_actor = self._create_cube()

        if self.viewer is not None:
            self.gym.viewer_camera_look_at(
                self.viewer,
                self.env,
                gymapi.Vec3(0.34, -0.32, 0.22),
                gymapi.Vec3(0.045, 0.0, 0.10),
            )

    def _create_hand(self) -> int:
        gymapi = self.gymapi
        path = urdf_path(self.side)
        asset_root = os.path.dirname(path)
        asset_file = os.path.basename(path)

        options = gymapi.AssetOptions()
        options.fix_base_link = True
        options.disable_gravity = True
        options.default_dof_drive_mode = gymapi.DOF_MODE_POS
        options.collapse_fixed_joints = False
        options.use_mesh_materials = True
        options.override_com = True
        options.override_inertia = True

        asset = self.gym.load_asset(self.sim, asset_root, asset_file, options)
        if asset is None:
            raise RuntimeError(f"failed to load asset: {path}")

        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(0.0, 0.0, 0.0)
        pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
        actor = self.gym.create_actor(self.env, asset, pose, "linkerhand", 0, 1)

        self.dof_names = list(self.gym.get_actor_dof_names(self.env, actor))
        self.num_dofs = len(self.dof_names)
        self.dof_name_to_i: Dict[str, int] = {n: i for i, n in enumerate(self.dof_names)}
        self.targets = np.zeros(self.num_dofs, dtype=np.float32)

        props = self.gym.get_actor_dof_properties(self.env, actor)
        props["driveMode"].fill(gymapi.DOF_MODE_POS)
        props["stiffness"].fill(float(self.args.stiffness))
        props["damping"].fill(float(self.args.damping))
        props["effort"].fill(float(self.args.effort))
        self.gym.set_actor_dof_properties(self.env, actor, props)

        self._print_mapping_once()
        return actor

    def _create_cube(self) -> int:
        gymapi = self.gymapi
        options = gymapi.AssetOptions()
        options.density = float(self.args.cube_density)
        options.disable_gravity = False
        cube_asset = self.gym.create_box(
            self.sim,
            float(self.args.cube_size),
            float(self.args.cube_size),
            float(self.args.cube_size),
            options,
        )
        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(float(self.args.cube_x),
                             float(self.args.cube_y),
                             float(self.args.cube_z))
        pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
        actor = self.gym.create_actor(self.env, cube_asset, pose, "cube_55mm", 0, 0)
        try:
            color = gymapi.Vec3(0.1, 0.45, 0.9)
            self.gym.set_rigid_body_color(self.env, actor, 0,
                                          gymapi.MESH_VISUAL_AND_COLLISION,
                                          color)
        except Exception:
            pass
        return actor

    def _print_mapping_once(self) -> None:
        missing = [
            JOINT_NAME[idx] for idx in ACTIVE_IDX
            if JOINT_NAME[idx] not in self.dof_name_to_i
        ]
        missing += [m for m in self.mimics if m not in self.dof_name_to_i]
        if missing:
            print(f"[isaac-gym] warning: missing DOFs in asset: {missing}", flush=True)
        print(f"[isaac-gym] loaded {self.num_dofs} DOFs", flush=True)

    def set_joints(self, joint_rad) -> None:
        q = np.asarray(joint_rad, dtype=float).reshape(-1)
        if q.shape[0] != N_JOINTS:
            raise ValueError(f"joint_rad must have {N_JOINTS} entries, got {q.shape[0]}")

        for idx in ACTIVE_IDX:
            name = JOINT_NAME[idx]
            dof_i = self.dof_name_to_i.get(name)
            if dof_i is not None:
                self.targets[dof_i] = float(q[idx])

        for mimic_name, (driver, mult, off) in self.mimics.items():
            mimic_i = self.dof_name_to_i.get(mimic_name)
            semantic_i = self.driver_name_to_semantic.get(driver)
            if mimic_i is None or semantic_i is None:
                continue
            self.targets[mimic_i] = float(mult * q[semantic_i] + off)

        self.gym.set_actor_dof_position_targets(self.env, self.hand_actor, self.targets)

    def step(self, n: int = 1) -> bool:
        for _ in range(max(1, int(n))):
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)
        if self.viewer is not None:
            if self.gym.query_viewer_has_closed(self.viewer):
                return False
            self.gym.step_graphics(self.sim)
            self.gym.draw_viewer(self.viewer, self.sim, True)
        return True

    def close(self) -> None:
        if getattr(self, "viewer", None) is not None:
            self.gym.destroy_viewer(self.viewer)
            self.viewer = None
        if getattr(self, "sim", None) is not None:
            self.gym.destroy_sim(self.sim)
            self.sim = None


def run(args) -> None:
    # Fail before opening the camera if Isaac Gym is missing or the Python
    # version is unsupported.
    model = IsaacGymLinkerHand(args)
    source, pipeline, detections = _make_stream(args)
    adjust = _compose_adjust(
        _thumb_adjuster(args.thumb_gain, args.thumb_cross_gain),
        _little_abd_adjuster(args.little_abd_gain),
        _thumb_grasp_adjuster(args.thumb_grasp_gain),
        _thumb_orient_adjuster(args.thumb_orient_gain),
        _thumb_only_adjuster(args.thumb_only),
    )

    cv2 = None
    if args.show_camera:
        import cv2  # noqa: F401

    prev_safe = None
    last_side = None
    frame_i = 0
    try:
        for det in detections:
            pf = pipeline.process(det)
            if pf is None:
                if not _camera_preview(cv2, source, "Isaac Gym camera", None):
                    break
                if not model.step(args.sim_steps_per_frame):
                    break
                continue

            if pf.side != last_side:
                prev_safe = None
                last_side = pf.side
            if not _camera_preview(cv2, source, "Isaac Gym camera", pf):
                break

            out = teleop_command(
                pf.landmarks,
                pf.side,
                prev_safe,
                DEFAULT_DT,
                use_filter=not args.no_filter,
                candidate_adjust=adjust,
            )
            if out["safe"] is not None:
                prev_safe = out["safe"]

            model.set_joints(out["command"])
            if not model.step(args.sim_steps_per_frame):
                break

            frame_i += 1
            if args.max_frames is not None and frame_i >= args.max_frames:
                break
            if args.realtime_sleep > 0:
                time.sleep(args.realtime_sleep)
    finally:
        source.close()
        model.close()
        if cv2 is not None:
            cv2.destroyAllWindows()


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="MediaPipe teleop grasp of a 5.5 cm cube in Isaac Gym.")
    ap.add_argument("--source", choices=("webcam", "realsense", "video"),
                    default="webcam")
    ap.add_argument("--camera-index", type=int, default=0)
    ap.add_argument("--video-path", default=None)
    ap.add_argument("--playback-rate", type=float, default=1.0)
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
    ap.add_argument("--headless", action="store_true")

    ap.add_argument("--cube-size", type=float, default=CUBE_SIZE_M)
    ap.add_argument("--cube-x", type=float, default=0.065)
    ap.add_argument("--cube-y", type=float, default=0.0)
    ap.add_argument("--cube-z", type=float, default=0.115)
    ap.add_argument("--cube-density", type=float, default=350.0)

    ap.add_argument("--gravity-x", type=float, default=-4.0,
                    help="palm-ward gravity helps grasp without an arm/wrist")
    ap.add_argument("--gravity-y", type=float, default=0.0)
    ap.add_argument("--gravity-z", type=float, default=0.0)
    ap.add_argument("--sim-hz", type=float, default=240.0)
    ap.add_argument("--substeps", type=int, default=2)
    ap.add_argument("--sim-steps-per-frame", type=int, default=8)
    ap.add_argument("--realtime-sleep", type=float, default=0.0)

    ap.add_argument("--stiffness", type=float, default=60.0)
    ap.add_argument("--damping", type=float, default=5.0)
    ap.add_argument("--effort", type=float, default=0.25)

    ap.add_argument("--thumb-gain", type=float, default=1.0)
    ap.add_argument("--thumb-cross-gain", type=float, default=0.0)
    ap.add_argument("--little-abd-gain", type=float, default=1.0)
    ap.add_argument("--thumb-grasp-gain", type=float, default=0.25)
    ap.add_argument("--thumb-orient-gain", type=float, default=0.6)
    ap.add_argument("--thumb-only", action="store_true")

    ap.add_argument("--compute-device-id", type=int, default=0)
    ap.add_argument("--graphics-device-id", type=int, default=0)
    return ap


def main(argv: Optional[list] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
