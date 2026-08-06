import json
from pathlib import Path

import numpy as np

from src.comms.action_library import (
    ActionLibrary,
    FEATURE_PROFILE_NO_FINGER_SPLAY,
    FEATURE_PROFILE_THUMB_LITTLE_CONTACT,
    LivePoseMatcher,
    OnlinePhaseMatcher,
    PHASE_MAPPING_MOTION,
    PHASE_MAPPING_THUMB_LITTLE_CONTACT,
    PHASE_MAPPING_THUMB_LITTLE_ROUNDTRIP,
    Primitive,
    StreamingMatcher,
    TrajectoryExecutor,
    dtw_distance,
    g20_range_to_sim_radians,
    interpolate_waypoints,
    landmark_feature,
    template_phase_axis,
)


def _hand() -> np.ndarray:
    points = np.zeros((21, 3), dtype=np.float32)
    roots = ((1, -0.8), (5, -0.4), (9, 0.0), (13, 0.4), (17, 0.8))
    tips = (4, 8, 12, 16, 20)
    for (root, x), tip in zip(roots, tips):
        points[root:tip + 1, 0] = x
        points[root:tip + 1, 2] = np.linspace(0.3, 1.5, tip - root + 1)
    return points


def _pose(value: float) -> list[float]:
    pose = [value] * 20
    for index in (11, 12, 13, 14):
        pose[index] = 255.0
    return pose


def test_landmark_feature_is_translation_and_scale_invariant():
    hand = _hand()
    expected = landmark_feature(hand)

    actual = landmark_feature(hand * 2.7 + np.asarray([4.0, -2.0, 8.0]))

    assert expected.shape == (83,)
    np.testing.assert_allclose(actual, expected, atol=1e-6)


def test_no_splay_feature_ignores_non_thumb_lateral_offsets():
    hand = _hand()
    splayed = hand.copy()
    for direction, chain in zip((-1.0, -0.5, 0.5, 1.0), (
        (5, 6, 7, 8),
        (9, 10, 11, 12),
        (13, 14, 15, 16),
        (17, 18, 19, 20),
    )):
        for step, landmark in enumerate(chain[1:], start=1):
            splayed[landmark, 1] += direction * 0.12 * step

    full = landmark_feature(hand)
    full_splayed = landmark_feature(splayed)
    flexion = landmark_feature(
        hand, feature_profile=FEATURE_PROFILE_NO_FINGER_SPLAY
    )
    flexion_splayed = landmark_feature(
        splayed, feature_profile=FEATURE_PROFILE_NO_FINGER_SPLAY
    )

    assert float(np.linalg.norm(full - full_splayed)) > 0.1
    np.testing.assert_allclose(flexion_splayed, flexion, atol=1e-6)


def test_thumb_little_contact_profile_keeps_only_contact_lateral_distance():
    hand = _hand()
    little_splayed = hand.copy()
    for step, landmark in enumerate((18, 19, 20), start=1):
        little_splayed[landmark, 1] += 0.2 * step

    baseline = landmark_feature(
        hand, feature_profile=FEATURE_PROFILE_THUMB_LITTLE_CONTACT
    )
    changed = landmark_feature(
        little_splayed, feature_profile=FEATURE_PROFILE_THUMB_LITTLE_CONTACT
    )

    assert abs(float(changed[77] - baseline[77])) > 0.05
    np.testing.assert_allclose(changed[:77], baseline[:77], atol=1e-6)
    np.testing.assert_allclose(changed[78:], baseline[78:], atol=1e-6)


def test_dtw_accepts_a_time_stretched_motion():
    short = np.asarray([[0.0], [0.5], [1.0], [0.5], [0.0]], dtype=np.float32)
    stretched = np.repeat(short, 2, axis=0)

    assert dtw_distance(short, stretched) < 1e-7
    assert dtw_distance(short, 1.0 - stretched) > 0.2


def test_waypoints_expand_to_sdk_trajectory_and_pin_reserved():
    trajectory = interpolate_waypoints([
        {"pose": _pose(255)},
        {"pose": _pose(200), "duration": 0.5},
    ], fps=10)

    assert trajectory.shape == (6, 20)
    assert trajectory[0, 0] == 255
    assert trajectory[-1, 0] == 200
    assert np.all(trajectory[:, 11:15] == 255)


def test_g20_open_pose_maps_to_zero_radian_preview():
    radians = g20_range_to_sim_radians([
        255, 255, 255, 255, 255, 255, 193, 148, 105, 42,
        245, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    ])

    np.testing.assert_allclose(radians, np.zeros(20), atol=1e-7)


def test_library_loads_manifest_arrays(tmp_path):
    folder = tmp_path / "primitive_000_close"
    folder.mkdir()
    np.save(folder / "robot_trajectory.npy", np.asarray([_pose(255), _pose(200)]))
    np.save(folder / "human_take_000.npy", np.stack([_hand()] * 8))
    (tmp_path / "manifest.json").write_text(json.dumps({
        "schema": ActionLibrary.SCHEMA,
        "fps": 30,
        "feature_profile": FEATURE_PROFILE_NO_FINGER_SPLAY,
        "primitives": [{
            "id": 0,
            "name": "close",
            "robot_trajectory": "primitive_000_close/robot_trajectory.npy",
            "human_templates": ["primitive_000_close/human_take_000.npy"],
            "phase_mapping": PHASE_MAPPING_MOTION,
            "phase_motion_epsilon": 0.006,
            "phase_endpoint_snap_distance": 0.012,
            "best_effort_spread_feedback": True,
            "max_command_lead": 28,
        }],
    }))

    library = ActionLibrary.load(tmp_path)

    assert library.primitives[0].trajectory.shape == (2, 20)
    assert library.primitives[0].templates[0].shape == (8, 83)
    assert library.feature_profile == FEATURE_PROFILE_NO_FINGER_SPLAY
    assert library.primitives[0].phase_mapping == PHASE_MAPPING_MOTION
    assert library.primitives[0].phase_motion_epsilon == 0.006
    assert library.primitives[0].phase_endpoint_snap_distance == 0.012
    assert library.primitives[0].phase_contact_fraction == 0.5
    assert library.primitives[0].best_effort_spread_feedback is True
    assert library.primitives[0].max_command_lead == 28
    np.testing.assert_allclose(
        library.primitives[0].templates[0][0],
        landmark_feature(
            _hand(), feature_profile=FEATURE_PROFILE_NO_FINGER_SPLAY
        ),
    )


def test_streaming_matcher_emits_confirmed_primitive():
    template = np.linspace(0.0, 1.0, 8, dtype=np.float32)[:, None]
    primitive = Primitive(3, "push", np.asarray([_pose(230)]), (template,), threshold=0.05)
    matcher = StreamingMatcher(
        ActionLibrary(Path("."), [primitive]),
        evaluation_interval=1,
        confirm_evaluations=1,
        margin=0.0,
    )

    result = None
    for feature in template:
        result = matcher.update(feature)
        if result is not None:
            break

    assert result is not None
    assert result.primitive_id == 3


def test_executor_blends_and_step_limits_against_observed_state():
    primitive = Primitive(
        4,
        "close",
        np.asarray([_pose(200), _pose(180)], dtype=np.float32),
        (np.zeros((6, 1), dtype=np.float32),),
    )
    library = ActionLibrary(Path("."), [primitive])
    executor = TrajectoryExecutor(library, max_step=5, blend_frames=2)
    assert executor.enqueue(4)

    first = executor.tick(_pose(255))
    second = executor.tick(first)

    assert first[0] == 250
    assert second[0] == 245
    assert first[11:15] == [255, 255, 255, 255]


def test_online_phase_matcher_locks_prefix_and_advances_monotonically():
    positive = np.linspace(0.0, 1.0, 20, dtype=np.float32)[:, None]
    negative = np.linspace(0.0, -1.0, 20, dtype=np.float32)[:, None]
    library = ActionLibrary(Path("."), [
        Primitive(1, "positive", np.asarray([_pose(220)]), (positive,), threshold=0.20),
        Primitive(2, "negative", np.asarray([_pose(210)]), (negative,), threshold=0.20),
    ])
    matcher = OnlinePhaseMatcher(
        library,
        min_lock_phase=0.15,
        lock_margin=0.02,
        confirm_frames=2,
        max_template_advance=2,
    )

    results = [matcher.update(frame) for frame in positive]
    locked = [result for result in results if result.locked]

    assert locked
    assert all(result.primitive_id == 1 for result in locked)
    assert all(right.phase >= left.phase for left, right in zip(locked, locked[1:]))
    assert locked[-1].phase > 0.9


def test_motion_phase_axis_compresses_and_snaps_a_stable_endpoint():
    template = np.asarray(
        [[0.0], [0.2], [0.4], [0.6], [0.8], [1.0], [1.002], [0.999], [1.001]],
        dtype=np.float32,
    )

    phases = template_phase_axis(
        template,
        mapping=PHASE_MAPPING_MOTION,
        motion_epsilon=0.01,
        endpoint_snap_distance=0.01,
        endpoint_window=3,
    )

    np.testing.assert_allclose(phases[5:], np.ones(4), atol=1e-7)
    assert np.all(np.diff(phases) >= 0.0)
    assert phases[0] == 0.0


def test_thumb_little_contact_phase_tracks_closing_distance_and_ignores_rebound():
    template = np.zeros((8, 83), dtype=np.float32)
    template[:, 77] = np.asarray(
        [2.0, 1.8, 1.5, 1.0, 0.55, 0.20, 0.15, 0.24], dtype=np.float32
    )

    phases = template_phase_axis(
        template,
        mapping=PHASE_MAPPING_THUMB_LITTLE_CONTACT,
        endpoint_window=2,
    )

    assert phases[0] == 0.0
    assert phases[3] > 0.45
    np.testing.assert_allclose(phases[5:], np.ones(3), atol=1e-7)
    assert np.all(np.diff(phases) >= 0.0)


def test_thumb_little_roundtrip_phase_maps_contact_to_configured_pivot():
    template = np.zeros((9, 83), dtype=np.float32)
    template[:, 77] = np.asarray(
        [2.0, 1.6, 1.1, 0.6, 0.2, 0.35, 0.55, 0.75, 0.9],
        dtype=np.float32,
    )

    phases = template_phase_axis(
        template,
        mapping=PHASE_MAPPING_THUMB_LITTLE_ROUNDTRIP,
        contact_fraction=0.70,
        endpoint_snap_distance=0.0,
    )

    assert phases[0] == 0.0
    assert phases[4] == np.float32(0.70)
    assert phases[-1] == 1.0
    assert np.all(np.diff(phases) >= 0.0)
    assert phases[3] < 0.70 < phases[5]


def test_live_pose_matcher_uses_motion_endpoint_snap():
    template = np.asarray(
        [[0.0], [0.2], [0.4], [0.6], [0.8], [1.0], [1.002], [0.999], [1.001]],
        dtype=np.float32,
    )
    primitive = Primitive(
        3,
        "partial_close",
        np.asarray([_pose(220)]),
        (template,),
        threshold=0.20,
        phase_mapping=PHASE_MAPPING_MOTION,
        phase_motion_epsilon=0.01,
        phase_endpoint_snap_distance=0.01,
        phase_endpoint_window=3,
    )
    matcher = LivePoseMatcher(
        ActionLibrary(Path("."), [primitive]),
        selected_id=3,
        confirm_frames=1,
        max_phase_step=1.0,
        phase_smoothing=1.0,
    )

    result = matcher.update(np.asarray([0.998], dtype=np.float32))

    assert result.locked
    assert result.phase == 1.0


def test_online_phase_matcher_does_not_lock_ambiguous_shared_prefix():
    shared = np.zeros((6, 1), dtype=np.float32)
    left = np.concatenate((shared, np.ones((8, 1), dtype=np.float32)), axis=0)
    right = np.concatenate((shared, -np.ones((8, 1), dtype=np.float32)), axis=0)
    library = ActionLibrary(Path("."), [
        Primitive(1, "left", np.asarray([_pose(220)]), (left,), threshold=0.25),
        Primitive(2, "right", np.asarray([_pose(210)]), (right,), threshold=0.25),
    ])
    matcher = OnlinePhaseMatcher(
        library,
        min_lock_phase=0.10,
        lock_margin=0.02,
        confirm_frames=1,
    )

    results = [matcher.update(frame) for frame in shared]

    assert not any(result.locked for result in results)


def test_live_pose_matcher_tracks_forward_and_reverse():
    positive = np.linspace(0.0, 1.0, 21, dtype=np.float32)[:, None]
    negative = np.linspace(0.0, -1.0, 21, dtype=np.float32)[:, None]
    library = ActionLibrary(Path("."), [
        Primitive(1, "positive", np.asarray([_pose(220)]), (positive,), threshold=0.20),
        Primitive(2, "negative", np.asarray([_pose(210)]), (negative,), threshold=0.20),
    ])
    matcher = LivePoseMatcher(
        library,
        confirm_frames=1,
        match_margin=0.02,
        max_phase_step=1.0,
        phase_smoothing=1.0,
    )

    forward = [matcher.update(frame) for frame in positive[4:]]
    reverse = [matcher.update(frame) for frame in positive[-2::-1]]

    assert all(result.locked and result.primitive_id == 1 for result in forward)
    assert forward[-1].phase > 0.95
    assert reverse[-1].phase < 0.05
    assert all(
        right.phase <= left.phase
        for left, right in zip(reverse, reverse[1:])
    )


def test_live_pose_matcher_switches_class_after_confirmation():
    positive = np.linspace(0.0, 1.0, 21, dtype=np.float32)[:, None]
    negative = np.linspace(0.0, -1.0, 21, dtype=np.float32)[:, None]
    library = ActionLibrary(Path("."), [
        Primitive(1, "positive", np.asarray([_pose(220)]), (positive,), threshold=0.20),
        Primitive(2, "negative", np.asarray([_pose(210)]), (negative,), threshold=0.20),
    ])
    matcher = LivePoseMatcher(
        library,
        confirm_frames=1,
        switch_confirm_frames=2,
        match_margin=0.02,
        max_phase_step=1.0,
        phase_smoothing=1.0,
    )
    assert matcher.update(np.asarray([0.8], dtype=np.float32)).primitive_id == 1

    first = matcher.update(np.asarray([-0.8], dtype=np.float32))
    second = matcher.update(np.asarray([-0.8], dtype=np.float32))

    assert not first.locked
    assert second.locked
    assert second.primitive_id == 2


def test_live_pose_matcher_class_bias_breaks_an_ambiguous_tie():
    shared = np.linspace(0.0, 1.0, 11, dtype=np.float32)[:, None]
    library = ActionLibrary(Path("."), [
        Primitive(2, "together", np.asarray([_pose(220)]), (shared,), threshold=0.20),
        Primitive(3, "spread", np.asarray([_pose(210)]), (shared,), threshold=0.20),
    ])
    matcher = LivePoseMatcher(
        library,
        confirm_frames=1,
        match_margin=0.01,
        max_phase_step=1.0,
        phase_smoothing=1.0,
    )

    result = matcher.update(
        np.asarray([0.5], dtype=np.float32),
        class_distance_bias={3: 0.025},
    )

    assert result.locked
    assert result.primitive_id == 2


def test_live_pose_matcher_hard_exclusion_removes_a_locked_class():
    positive = np.linspace(0.0, 1.0, 11, dtype=np.float32)[:, None]
    negative = np.linspace(0.0, -1.0, 11, dtype=np.float32)[:, None]
    library = ActionLibrary(Path("."), [
        Primitive(2, "together", np.asarray([_pose(220)]), (negative,), threshold=2.0),
        Primitive(3, "spread", np.asarray([_pose(210)]), (positive,), threshold=2.0),
    ])
    matcher = LivePoseMatcher(
        library,
        confirm_frames=1,
        match_margin=0.0,
        max_phase_step=1.0,
        phase_smoothing=1.0,
    )
    assert matcher.update(np.asarray([0.8], dtype=np.float32)).primitive_id == 3

    result = matcher.update(
        np.asarray([0.8], dtype=np.float32),
        excluded_ids={3},
    )

    assert result.locked
    assert result.primitive_id == 2
