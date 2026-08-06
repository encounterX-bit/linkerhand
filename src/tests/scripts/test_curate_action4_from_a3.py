import numpy as np

from pathlib import Path

from scripts.curate_action4_from_a3 import (
    StableRun,
    curate_episode,
    first_run_after,
    stable_runs,
)


def test_stable_runs_keeps_only_runs_meeting_minimum_length():
    assert stable_runs([0, 1, 1, 0, 1, 1, 1, 0], 3) == [StableRun(4, 6)]


def test_first_run_after_clips_a_run_to_requested_start():
    run = first_run_after(np.array([0, 1, 1, 1, 1, 0]), 2, 3)
    assert run is not None
    assert (run.start, run.end) == (3, 4)


def test_curated_fragment_excludes_stable_endpoint_observations():
    a3 = np.zeros(20, dtype=np.float32)
    left = np.zeros(20, dtype=np.float32)
    left[[0, 5, 10, 15]] = [10, 20, 30, 40]
    endpoint = np.zeros(20, dtype=np.float32)
    endpoint[[0, 5, 10, 15]] = [50, 60, 70, 80]
    states = (
        [a3.copy() for _ in range(5)]
        + [np.array([*left[:15], 0, *left[16:]]) for _ in range(3)]
        + [left.copy() for _ in range(3)]
        + [endpoint.copy() for _ in range(5)]
    )
    rows = tuple(
        {"joint_pos": state.tolist(), "last_action": state.tolist()}
        for state in states
    )

    curated, reason = curate_episode(
        Path("episode"),
        rows,
        a3_endpoint=a3,
        a4_left=left,
        a4_endpoint=endpoint,
        start_tolerance=0,
        left_tolerance=0,
        close_tolerance=0,
        endpoint_tolerance=0,
        start_confirm_frames=5,
        stage_confirm_frames=3,
        endpoint_confirm_frames=5,
    )

    assert reason == "kept"
    assert curated is not None
    assert curated.endpoint_frame == 11
    assert curated.end_frame == 10
