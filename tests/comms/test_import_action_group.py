import json
from types import SimpleNamespace

import numpy as np

from src.comms.action_library import ActionLibrary, G20_OPEN_POSE
from src.comms.import_action_group import import_group


def _landmarks(frame, take):
    points = np.zeros((21, 3), dtype=np.float32)
    for index in range(21):
        points[index] = [index * 0.01, (index % 5) * 0.02, index * 0.003]
    points[8:, 1] += frame * 0.002 + take * 0.0005
    return points.tolist()


def _group(tmp_path, *, state_error=2):
    group = tmp_path / "group_000"
    (group / "human").mkdir(parents=True)
    (group / "robot").mkdir()
    takes = []
    rows = []
    for take in range(2):
        start = len(rows)
        for frame in range(6):
            rows.append({
                "index": len(rows),
                "fresh": True,
                "landmarks_hand_base": _landmarks(frame, take),
            })
        takes.append({
            "take_index": take,
            "start_sample": start,
            "end_sample": len(rows),
            "frames": 6,
        })
    (group / "human" / "samples.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    (group / "group.json").write_text(
        json.dumps({
            "status": "complete",
            "human_takes": takes,
            "robot_waypoints": 2,
        }),
        encoding="utf-8",
    )
    first = G20_OPEN_POSE.astype(float).tolist()
    second = list(first)
    second[1] = 100
    state = list(second)
    state[1] += state_error
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


def _args(group, library):
    return SimpleNamespace(
        group=group,
        library=library,
        primitive_id=0,
        name="test_motion",
        fps=30.0,
        trajectory_max_step=5,
        max_recorded_state_error=10.0,
        threshold=None,
        cooldown_frames=20,
        replace=False,
    )


def test_import_audited_group_builds_loadable_library(tmp_path):
    library_path = tmp_path / "library"
    result = import_group(_args(_group(tmp_path), library_path))
    library = ActionLibrary.load(library_path)

    assert result["templates"] == 2
    assert result["trajectory_frames"] > 16
    assert 0.06 <= result["threshold"] <= 0.18
    assert len(library.primitives[0].templates) == 2
    assert np.max(np.abs(np.diff(library.primitives[0].trajectory[:, 1]))) <= 5


def test_import_rejects_group_with_large_recorded_state_error(tmp_path):
    args = _args(_group(tmp_path, state_error=20), tmp_path / "library")

    with np.testing.assert_raises_regex(ValueError, "preflight blocked"):
        import_group(args)


def test_import_can_audit_spread_coupling_separately(tmp_path):
    group = _group(tmp_path)
    waypoint_path = group / "robot" / "waypoints.json"
    payload = json.loads(waypoint_path.read_text(encoding="utf-8"))
    payload["waypoints"][1]["state"][9] += 40
    waypoint_path.write_text(json.dumps(payload), encoding="utf-8")
    args = _args(group, tmp_path / "library")
    args.allow_spread_coupling_error = True
    args.max_recorded_spread_error = 45.0
    args.require_static_thumb = True
    args.max_thumb_command_span = 2.0

    result = import_group(args)
    manifest = json.loads((result["library"] / "manifest.json").read_text())
    record = manifest["primitives"][0]

    assert record["source_max_recorded_primary_error"] == 2
    assert record["source_max_recorded_spread_error"] == 40
    assert record["source_thumb_command_span"] == [0, 0, 0, 0]


def test_import_can_pair_human_and_robot_from_separate_groups(tmp_path):
    human_group = _group(tmp_path / "human_source")
    robot_group = _group(tmp_path / "robot_source")
    # Prove the importer does not accidentally read the robot group's human data.
    (robot_group / "human" / "samples.jsonl").write_text("", encoding="utf-8")
    args = _args(None, tmp_path / "library")
    args.human_group = human_group
    args.robot_group = robot_group

    result = import_group(args)
    manifest = json.loads((result["library"] / "manifest.json").read_text())
    record = manifest["primitives"][0]

    assert result["templates"] == 2
    assert record["human_source_group"] == str(human_group.resolve())
    assert record["robot_source_group"] == str(robot_group.resolve())
    assert record["source_group"] == str(robot_group.resolve())
