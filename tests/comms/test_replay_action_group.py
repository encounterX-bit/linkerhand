import json

import numpy as np

from src.comms.action_library import ACTIVE_IDX, G20_OPEN_POSE, RESERVED_IDX
from src.comms.replay_action_group import (
    densify_trajectory,
    load_replay_group,
    main,
    playback_trajectory,
)


def _write_group(tmp_path, *, error=2.0, status="complete"):
    group = tmp_path / "group_000"
    (group / "robot").mkdir(parents=True)
    first = G20_OPEN_POSE.astype(float).tolist()
    second = G20_OPEN_POSE.astype(float).tolist()
    second[1] = 100.0
    state = list(second)
    state[1] += error
    (group / "group.json").write_text(
        json.dumps({"status": status, "robot_waypoints": 2}), encoding="utf-8"
    )
    (group / "robot" / "waypoints.json").write_text(
        json.dumps({
            "trajectory_waypoints": [
                {"pose": first},
                {"pose": second, "duration": 0.5},
            ],
            "waypoints": [
                {"command": first, "state": first},
                {"command": second, "state": state},
            ],
        }),
        encoding="utf-8",
    )
    return group


def test_replay_group_preflight_accepts_small_recorded_error(tmp_path):
    group = load_replay_group(_write_group(tmp_path), max_recorded_state_error=10)

    assert group.issues == ()
    assert group.recorded_waypoints == 2
    assert group.max_recorded_error == 2
    assert group.worst_waypoint == 1
    assert group.worst_joint == 1


def test_replay_group_preflight_blocks_large_error_and_hardware_main(tmp_path, monkeypatch):
    path = _write_group(tmp_path, error=20)
    group = load_replay_group(path, max_recorded_state_error=10)

    assert any("exceeds limit" in issue for issue in group.issues)
    monkeypatch.setenv("HW_ENABLE_TOKEN", "1")
    assert main(["--group", str(path), "--enable-motion"]) == 2


def test_densify_and_blend_respect_active_step_and_reserved_values():
    end = G20_OPEN_POSE.copy()
    end[1] = 0
    end[16] = 10
    frames = playback_trajectory(
        np.stack((end,)),
        start_pose=G20_OPEN_POSE,
        max_step=5,
        blend_frames=8,
    )

    deltas = np.abs(np.diff(frames[:, list(ACTIVE_IDX)], axis=0))
    assert float(np.max(deltas)) <= 5.0 + 1e-5
    assert np.all(frames[:, list(RESERVED_IDX)] == 255)
    np.testing.assert_allclose(frames[-1], end)


def test_densify_rejects_nonpositive_step():
    with np.testing.assert_raises(ValueError):
        densify_trajectory(np.stack((G20_OPEN_POSE,)), max_step=0)


def test_default_main_is_hardware_free_even_when_preflight_blocked(tmp_path, capsys):
    path = _write_group(tmp_path, error=20)

    assert main(["--group", str(path), "--print-every", "1000"]) == 0
    output = capsys.readouterr().out
    assert "BLOCKED" in output
    assert "DRY RUN only" in output
    assert "no ROS publisher created" in output
