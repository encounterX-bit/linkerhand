import json
from types import SimpleNamespace

import numpy as np

from src.comms.analyze_action_groups import marker_segments, motion_segments
from src.comms.group_action_recorder import (
    GroupCapture,
    GroupWorkflow,
    Phase,
    draw_hand_overlay,
    parse_args,
)


def test_group_workflow_cycles_human_robot_and_next_group():
    workflow = GroupWorkflow(start_index=7)

    assert workflow.space().action == "start_group"
    assert workflow.phase == Phase.HUMAN_READY
    assert workflow.space().action == "need_human_take"
    assert workflow.toggle_human_take().action == "start_take"
    assert workflow.phase == Phase.HUMAN_RECORDING
    assert workflow.space().action == "need_stop_take"
    assert workflow.toggle_human_take().action == "stop_take"
    assert workflow.human_takes == 1
    assert workflow.space().action == "stop_human"
    assert workflow.phase == Phase.ROBOT
    assert workflow.space().action == "need_waypoint"
    assert workflow.group_index == 7
    assert workflow.waypoint()
    event = workflow.space()

    assert event.action == "finalize_and_start"
    assert event.group_index == 7
    assert workflow.group_index == 8
    assert workflow.phase == Phase.HUMAN_READY


def test_m_toggle_records_many_takes_in_one_group():
    workflow = GroupWorkflow()
    workflow.space()

    for _ in range(3):
        assert workflow.toggle_human_take().action == "start_take"
        assert workflow.toggle_human_take().action == "stop_take"

    assert workflow.human_takes == 3
    assert workflow.phase == Phase.HUMAN_READY
    assert workflow.group_index == 0


def test_retry_transitions_keep_current_group():
    workflow = GroupWorkflow(start_index=2)
    workflow.space()
    workflow.toggle_human_take()
    workflow.toggle_human_take()
    workflow.space()
    workflow.waypoint()

    workflow.redo_human()
    assert workflow.phase == Phase.HUMAN_READY
    assert workflow.human_takes == 0
    assert workflow.robot_waypoints == 1
    assert workflow.group_index == 2

    workflow.redo_robot()
    assert workflow.phase == Phase.ROBOT
    assert workflow.robot_waypoints == 0
    assert workflow.group_index == 2


def test_group_capture_writes_human_and_robot_artifacts(tmp_path):
    capture = GroupCapture(tmp_path, 0, jpeg_quality=90)
    image = np.zeros((32, 48, 3), dtype=np.uint8)
    landmarks = np.arange(63, dtype=np.float32).reshape(21, 3)
    processed = SimpleNamespace(
        detected=True, held=False, score=0.9, side="right", landmarks=landmarks
    )
    source = SimpleNamespace(
        last_world_landmarks_raw=landmarks,
        last_landmarks_raw_px=np.zeros((21, 2), dtype=np.float32),
    )

    capture.start_human_take()
    assert capture.add_human(image, processed, source)
    capture.finish_human_take()
    capture.stop_human()
    capture.add_robot_waypoint(image, {
        "command": [255.0] * 20,
        "state": [254.0] * 20,
        "command_age_seconds": 0.01,
        "state_age_seconds": 0.02,
    }, suggested_duration=0.5)
    capture.finalize()

    metadata = json.loads((capture.path / "group.json").read_text())
    waypoints = json.loads((capture.path / "robot" / "waypoints.json").read_text())
    assert metadata["human_samples"] == 1
    assert metadata["human_fresh_samples"] == 1
    assert metadata["human_takes"][0]["frames"] == 1
    assert metadata["robot_waypoints"] == 1
    assert waypoints["trajectory_waypoints"][0]["pose"] == [255.0] * 20
    assert (capture.path / "human" / "images" / "000000.jpg").is_file()
    assert (capture.path / "robot" / "images" / "waypoint_000.jpg").is_file()


def test_group_capture_retries_archive_old_human_and_robot_data(tmp_path):
    capture = GroupCapture(tmp_path, 0, jpeg_quality=90)
    image = np.zeros((20, 20, 3), dtype=np.uint8)
    processed = SimpleNamespace(
        detected=True, held=False, score=0.9, side="right",
        landmarks=np.zeros((21, 3), dtype=np.float32),
    )
    source = SimpleNamespace(last_world_landmarks_raw=None, last_landmarks_raw_px=None)
    capture.start_human_take()
    capture.add_human(image, processed, source)
    capture.finish_human_take()
    capture.add_robot_waypoint(image, {
        "command": [255.0] * 20,
        "state": [255.0] * 20,
        "command_age_seconds": 0.01,
        "state_age_seconds": 0.01,
    }, suggested_duration=0.5)

    human_revision = capture.reset_human()
    robot_revision = capture.reset_robot()

    assert (human_revision / "samples.jsonl").is_file()
    assert (human_revision / "images" / "000000.jpg").is_file()
    assert (robot_revision / "waypoints.json").is_file()
    assert (robot_revision / "images" / "waypoint_000.jpg").is_file()
    assert capture.sample_count == 0
    assert capture.waypoints == []
    capture.finalize(status="incomplete")


def test_motion_segments_splits_repetitions_separated_by_long_pause():
    values = np.zeros((40, 2), dtype=np.float32)
    values[3:8, 0] = np.linspace(0.0, 1.0, 5)
    values[8:13, 0] = np.linspace(1.0, 0.0, 5)
    values[25:30, 1] = np.linspace(0.0, 1.0, 5)
    values[30:35, 1] = np.linspace(1.0, 0.0, 5)

    segments = motion_segments(
        values, threshold=0.05, pause_frames=4, pad_frames=1, min_frames=4
    )

    assert len(segments) == 2
    assert segments[0][1] < segments[1][0]


def test_marker_segments_maps_raw_markers_to_fresh_indices():
    sample_indices = [0, 1, 3, 4, 7, 8, 10, 11]
    markers = [{"sample_index": 7}]

    assert marker_segments(sample_indices, markers, min_frames=2) == [(0, 4), (4, 8)]


def test_hand_overlay_draws_preview_without_modifying_raw_frame():
    raw = np.zeros((100, 100, 3), dtype=np.uint8)
    points = np.stack((np.linspace(10, 90, 21), np.linspace(90, 10, 21)), axis=1)

    preview = draw_hand_overlay(raw, points, fresh=True)

    assert np.count_nonzero(preview) > 0
    assert np.count_nonzero(raw) == 0


def test_minimal_overlay_cli_flag():
    assert parse_args([]).minimal_overlay is False
    assert parse_args(["--minimal-overlay"]).minimal_overlay is True
