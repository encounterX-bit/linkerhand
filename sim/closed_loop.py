"""Closed-loop teleop sim: landmarks -> retarget() -> safety.filter() -> command
   -> step -> read, with CAN/control latency modeled as a delay buffer.

This is the G2 control loop. Each *control tick* (camera rate, 30 Hz):

    landmarks --retarget()--> candidate --safety.filter()--> safe command
        --[latency delay buffer]--> applied to PD motors --step physics--> read back

The retargeter (``finger_retarget.retarget``) and the safety guard
(``safety.filter``) are imported READ-ONLY — this module never modifies them. The
metric (segment-direction error) is taken from the achieved physical config via the
``src/kinematics`` FK authority, NOT from PyBullet.

LATENCY is modeled as an integer number of control frames of pure transport delay:
a command computed at tick k is applied at tick k+``latency_frames``. The buffer is a
deque; it is modeled delay, NOT compute, so it does not count against the loop-rate
budget (ticket Test 1).

TIMING: ``compute_us`` per tick = retarget + filter + sim-step (the real-time work).
The loop-rate gate is on the p99 of this, NOT including the modeled latency.
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from src.finger_retarget import retarget          # read-only system under test
from src import safety                              # read-only G2 guard

from .conventions import FINGER_ORDER, N_JOINTS
from .dynamics import L20Dynamics, PDGains, DEFAULT_GAINS
from .pipeline import human_segments, geodesic_angle


@dataclass
class TickRecord:
    frame: int
    compute_us: float          # retarget + filter + step (real-time work)
    t_retarget_us: float
    t_filter_us: float
    t_step_us: float
    applied: list              # the 20-vector actually sent to the motors this tick
    achieved: np.ndarray       # achieved 20-vector (from physics)
    filter_modified: bool
    filter_reason: object
    seg_err: dict              # {finger: (e_prox, e_dist)} achieved-vs-human, rad
    max_seg_err: float


class ClosedLoopSim:
    """Run the teleop loop over a landmark stream under dynamics + latency.

    Parameters
    ----------
    side : 'right' | 'left'
    latency_s : transport delay applied to every command (modeled, not compute).
    control_hz : control/camera rate (the loop period).
    sim_hz : physics step rate (>= control_hz; integer ratio).
    use_filter : route candidates through ``safety.filter`` (G2). When False the
        raw retarget output is commanded (the ablation baseline).
    """

    def __init__(self, side: str = "right", *, latency_s: float = 0.0,
                 control_hz: float = 30.0, sim_hz: float = 240.0,
                 gains: PDGains = DEFAULT_GAINS, use_filter: bool = True,
                 gravity=(0.0, 0.0, -9.81), dyn: L20Dynamics | None = None):
        self.side = side
        self.control_period = 1.0 / control_hz
        self.substeps = max(1, int(round(sim_hz / control_hz)))
        self.latency_frames = int(round(latency_s * control_hz))
        self.use_filter = use_filter
        self.dyn = dyn or L20Dynamics(side, timestep=1.0 / sim_hz, gains=gains,
                                      gravity=gravity)
        self._buffer = deque()           # pending commands (latency)
        self._prev_safe = None           # for the safety rate-limit guard
        self.frame = 0

    def close(self):
        self.dyn.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # -- one control tick -------------------------------------------------- #
    def tick(self, landmarks) -> TickRecord:
        # 1. retarget (timed)
        t0 = time.perf_counter()
        cand = retarget(landmarks, side=self.side)
        t_ret = time.perf_counter() - t0

        # 2. safety filter (timed); read-only G2 guard
        t_filt = 0.0
        modified, reason = False, None
        if self.use_filter:
            t1 = time.perf_counter()
            safe = safety.filter(cand, self._prev_safe, self.control_period,
                                 side=self.side)
            t_filt = time.perf_counter() - t1
            command = safe["joint_rad"]
            modified, reason = safe["modified"], safe["reason"]
            self._prev_safe = safe
        else:
            command = list(cand["joint_rad"])

        # 3. latency: enqueue this command; apply the one delayed by latency_frames.
        #    (deque pop — modeled transport delay, intentionally NOT timed.)
        self._buffer.append(command)
        if len(self._buffer) > self.latency_frames:
            self.dyn.set_command(self._buffer.popleft())

        # 4. step physics one control period (timed)
        t2 = time.perf_counter()
        self.dyn.step(self.substeps)
        t_step = time.perf_counter() - t2

        # 5. read achieved config + metric error (FK authority, not pybullet)
        achieved = self.dyn.achieved_joint_rad()
        robot = self.dyn.segment_dirs()
        human = human_segments(landmarks)
        seg_err = {f: (geodesic_angle(robot[f][0], human[f][0]),
                       geodesic_angle(robot[f][1], human[f][1]))
                   for f in FINGER_ORDER}
        max_err = max(max(e) for e in seg_err.values())

        rec = TickRecord(
            frame=self.frame,
            compute_us=(t_ret + t_filt + t_step) * 1e6,
            t_retarget_us=t_ret * 1e6, t_filter_us=t_filt * 1e6,
            t_step_us=t_step * 1e6,
            applied=self.dyn._cmd.tolist(), achieved=achieved,
            filter_modified=modified, filter_reason=reason,
            seg_err=seg_err, max_seg_err=max_err)
        self.frame += 1
        return rec

    def run(self, frames) -> list:
        """Run the loop over an iterable of landmark arrays; return [TickRecord]."""
        return [self.tick(lm) for lm in frames]
