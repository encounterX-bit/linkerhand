"""src/sim — G1 kinematic tracking harness for the Linker Hand L20.

Loads the vendored L20 URDF in PyBullet (KINEMATIC ONLY: resetJointState, no
dynamics), drives it with the read-only closed-form solver, and measures how well
the achieved finger-segment directions track the human's (ADR-0003 convention).

Public surface:
    L20Kinematics  — PyBullet FK harness (manual mimic enforcement)
    track_frame    — landmarks -> solve -> FK -> per-segment geodesic error
    human_segments, geodesic_angle
"""
from .kinematics import L20Kinematics, urdf_path
from .pipeline import track_frame, human_segments, geodesic_angle, error_rows
from .conventions import (
    ACTIVE_IDX, RESERVED_IDX, N_JOINTS, FINGER_ORDER,
)
# G2 dynamic harness (PyBullet dynamics/contact only; metric FK stays in src/kinematics)
from .dynamics import L20Dynamics, PDGains, DEFAULT_GAINS
from .closed_loop import ClosedLoopSim, TickRecord
from .grasp import (
    GraspScenario, GraspResult, run_grasp, SCENARIOS, CYLINDER, SPHERE,
)

__all__ = [
    "L20Kinematics", "urdf_path",
    "track_frame", "human_segments", "geodesic_angle", "error_rows",
    "ACTIVE_IDX", "RESERVED_IDX", "N_JOINTS", "FINGER_ORDER",
    # G2 dynamic
    "L20Dynamics", "PDGains", "DEFAULT_GAINS",
    "ClosedLoopSim", "TickRecord",
    "GraspScenario", "GraspResult", "run_grasp", "SCENARIOS", "CYLINDER", "SPHERE",
]
