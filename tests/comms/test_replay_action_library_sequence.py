from pathlib import Path

import numpy as np

from src.comms.action_library import (
    ACTIVE_IDX,
    ActionLibrary,
    G20_OPEN_POSE,
    Primitive,
    RESERVED_IDX,
)
from src.comms.replay_action_library_sequence import (
    build_sequence_segments,
    effective_command_lead_limit,
    largest_following_error,
    main,
    parse_args,
    parse_order,
    primitive_id_from_key,
    trajectory_following_indices,
)


def _pose(**changes: float) -> np.ndarray:
    pose = G20_OPEN_POSE.copy()
    for key, value in changes.items():
        pose[int(key)] = value
    return pose


def _library() -> ActionLibrary:
    feature = (np.zeros((2, 1), dtype=np.float32),)
    return ActionLibrary(Path("."), [
        Primitive(1, "one", np.stack((_pose(), _pose(**{"0": 100}))), feature),
        Primitive(2, "two", np.stack((_pose(), _pose(**{"1": 80}))), feature),
        Primitive(3, "three", np.stack((_pose(**{"1": 80}), _pose(**{"2": 60}))), feature),
    ])


def test_parse_order_preserves_explicit_sequence():
    assert parse_order("1, 3,2,3") == (1, 3, 2, 3)


def test_sequence_resets_open_by_default_and_can_be_disabled():
    default = parse_args(["--library", "library"])
    disabled = parse_args(["--library", "library", "--no-reset-before-sequence"])

    assert default.reset_before_sequence is True
    assert default.clear_faults_before_reset is True
    assert default.order == (1, 2, 3, 4, 5)
    assert default.current_limit == 100
    assert default.speed_limit == 100
    assert default.thumb_current_limit is None
    assert default.thumb_roundtrip_key == 6
    assert default.thumb_roundtrip_source_action == 2
    assert disabled.reset_before_sequence is False


def test_number_keys_select_only_available_primitive_ids():
    available = (1, 2, 3, 4, 5)

    assert primitive_id_from_key(ord("1"), available) == 1
    assert primitive_id_from_key(ord("5"), available) == 5
    assert primitive_id_from_key(ord("6"), available) is None
    assert primitive_id_from_key(ord(" "), available) is None


def test_largest_following_error_reports_thumb_tip_channel():
    state = G20_OPEN_POSE.copy()
    state[15] = 48

    error, index = largest_following_error(state, G20_OPEN_POSE, ACTIVE_IDX)

    assert error == 207
    assert index == 15


def test_sequence_segments_transition_safely_and_end_at_each_action():
    library = _library()
    segments = build_sequence_segments(
        library,
        (1, 2, 3),
        start_pose=G20_OPEN_POSE,
        max_step=10,
        blend_frames=2,
    )

    previous = G20_OPEN_POSE
    assert [item.primitive_id for item in segments] == [1, 2, 3]
    for segment in segments:
        combined = np.concatenate((previous[None, :], segment.frames), axis=0)
        deltas = np.abs(np.diff(combined[:, list(ACTIVE_IDX)], axis=0))
        assert float(np.max(deltas)) <= 10.0 + 1e-5
        assert np.all(segment.frames[:, list(RESERVED_IDX)] == 255)
        np.testing.assert_allclose(
            segment.frames[-1], library.primitives[segment.primitive_id].trajectory[-1]
        )
        previous = segment.frames[-1]


def test_sequence_segments_reject_missing_id():
    with np.testing.assert_raises(KeyError):
        build_sequence_segments(
            _library(),
            (1, 9),
            start_pose=G20_OPEN_POSE,
            max_step=10,
            blend_frames=2,
        )


def test_hardware_requires_exact_human_token(tmp_path, monkeypatch):
    monkeypatch.delenv("HW_ENABLE_TOKEN", raising=False)
    assert main([
        "--library", str(tmp_path / "missing"), "--enable-motion"
    ]) == 2


def test_command_lead_must_be_below_hard_following_limit(tmp_path):
    assert main([
        "--library",
        str(tmp_path / "missing"),
        "--max-command-lead",
        "40",
        "--max-following-error",
        "35",
    ]) == 2


def test_action_specific_contact_lead_keeps_hard_stop_headroom():
    assert effective_command_lead_limit(18, None, 35) == 18
    assert effective_command_lead_limit(18, 28, 35) == 28
    assert effective_command_lead_limit(18, 50, 35) == 34


def test_following_indices_ignore_only_static_four_finger_spreads():
    start = G20_OPEN_POSE.copy()
    end = start.copy()
    end[1] = 100
    end[7] = 120

    indices = trajectory_following_indices(np.stack((start, end)))

    assert 1 in indices
    assert 7 in indices
    assert 6 not in indices
    assert 8 not in indices
    assert 9 not in indices


def test_following_indices_can_make_all_coupled_spreads_best_effort():
    start = G20_OPEN_POSE.copy()
    end = start.copy()
    end[1] = 100
    end[6:10] = [125, 129, 125, 130]

    indices = trajectory_following_indices(
        np.stack((start, end)),
        include_moving_spreads=False,
    )

    assert 1 in indices
    assert all(index not in indices for index in (6, 7, 8, 9))
