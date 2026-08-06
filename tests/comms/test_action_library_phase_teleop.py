import numpy as np

from src.comms.action_library import (
    ACTIVE_IDX,
    G20_OPEN_POSE,
    PHASE_MAPPING_THUMB_LITTLE_CONTACT,
    PHASE_MAPPING_THUMB_LITTLE_ROUNDTRIP,
    RESERVED_IDX,
    Primitive,
    PhaseMatchResult,
)
from src.comms.action_library_phase_teleop import (
    Action23FingerSpreadRouter,
    Action3ContactAssist,
    action3_assist_may_acquire,
    Action4ThumbTipGate,
    THUMB_IDX,
    absolute_thumb_pose,
    action_thumb_mediapipe_fingers_target,
    active_pose_error,
    full_mediapipe_g20_target,
    frozen_thumb_target,
    four_finger_spread_score,
    hybrid_finger_target,
    is_delete_last_episode_key,
    is_full_teleop_toggle_key,
    is_manual_reset_key,
    is_thumb_freeze_toggle_key,
    main,
    manual_action_starts_from_first_frame,
    mediapipe_finger_fallback_target,
    nearest_trajectory_suffix,
    nonthumb_radians_to_g20_target,
    offset_thumb_pose,
    parse_args,
    reset_completion,
    selected_manual_trajectory,
    state_guarded_command,
    step_limited_command,
    thumb_roundtrip_trajectory,
    trajectory_suffix,
    trajectory_target,
)


def _contact_primitive() -> tuple[Primitive, np.ndarray]:
    template = np.zeros((11, 83), dtype=np.float32)
    template[:, 77] = np.linspace(2.0, 0.2, len(template), dtype=np.float32)
    primitive = Primitive(
        3,
        "thumb_contact",
        np.stack((G20_OPEN_POSE, G20_OPEN_POSE)),
        (template,),
        threshold=0.02,
        phase_mapping=PHASE_MAPPING_THUMB_LITTLE_CONTACT,
        phase_endpoint_window=1,
    )
    return primitive, template


def test_a3_contact_assist_acquires_and_tracks_both_directions():
    primitive, template = _contact_primitive()
    assist = Action3ContactAssist(
        primitive,
        activate_phase=0.08,
        release_phase=0.02,
        confirm_frames=2,
        threshold_scale=1.2,
        max_phase_step=1.0,
        phase_smoothing=1.0,
    )

    assert assist.update(template[-1]) is None
    contact = assist.update(template[-1])
    assert contact is not None and contact.phase == 1.0
    reverse = [assist.update(frame) for frame in template[-2::-1]]
    reverse_phases = [result.phase for result in reverse if result is not None]
    assert reverse_phases[-1] == 0.0
    assert all(
        right <= left for left, right in zip(reverse_phases, reverse_phases[1:])
    )

    assist.reset()
    forward = [assist.update(frame) for frame in template]
    forward_phases = [result.phase for result in forward if result is not None]
    assert forward_phases
    assert forward_phases[-1] == 1.0
    assert all(
        right >= left for left, right in zip(forward_phases, forward_phases[1:])
    )


def test_a3_contact_assist_rejects_contact_scalar_with_wrong_full_pose():
    primitive, template = _contact_primitive()
    assist = Action3ContactAssist(
        primitive, confirm_frames=1, threshold_scale=1.2
    )
    wrong = np.full(83, 4.0, dtype=np.float32)
    wrong[77] = template[-1, 77]

    assert assist.update(wrong) is None
    assert assist.active is False


def test_a3_contact_assist_yields_to_better_competing_action():
    primitive, template = _contact_primitive()
    query = template[-1].copy()
    query[0] = 0.1
    competitor = Primitive(
        2,
        "competitor",
        np.stack((G20_OPEN_POSE, G20_OPEN_POSE)),
        (query[None, :],),
        threshold=0.02,
    )
    assist = Action3ContactAssist(
        primitive,
        competitors=(competitor,),
        competition_slack=0.005,
        confirm_frames=1,
        threshold_scale=1.2,
    )

    assert assist.update(query) is None
    assert assist.active is False


def test_a3_contact_assist_cannot_steal_locked_non_a3_action():
    locked_a2 = PhaseMatchResult(2, "a2", 1.0, 0.01, 0.1, True)
    locked_a3 = PhaseMatchResult(3, "a3", 0.5, 0.01, 0.1, True)
    candidate_a3 = PhaseMatchResult(3, "a3", 0.2, 0.02, 0.1, False)

    assert action3_assist_may_acquire(None)
    assert not action3_assist_may_acquire(locked_a2)
    assert action3_assist_may_acquire(locked_a3)
    assert action3_assist_may_acquire(candidate_a3)


def test_a3_contact_assist_maps_inward_contact_then_outward_return():
    template = np.zeros((9, 83), dtype=np.float32)
    template[:, 77] = np.asarray(
        [2.0, 1.6, 1.1, 0.6, 0.2, 0.35, 0.55, 0.75, 0.9],
        dtype=np.float32,
    )
    primitive = Primitive(
        3,
        "thumb_roundtrip",
        np.stack((G20_OPEN_POSE, G20_OPEN_POSE)),
        (template,),
        threshold=0.02,
        phase_mapping=PHASE_MAPPING_THUMB_LITTLE_ROUNDTRIP,
        phase_endpoint_window=1,
        phase_contact_fraction=0.70,
    )
    assist = Action3ContactAssist(
        primitive,
        confirm_frames=1,
        threshold_scale=1.2,
        max_phase_step=1.0,
        phase_smoothing=1.0,
    )

    results = [assist.update(frame) for frame in template]
    phases = [result.phase for result in results if result is not None]

    assert phases
    assert phases[-1] == 1.0
    assert any(abs(phase - 0.70) < 1e-6 for phase in phases)
    assert all(right >= left for left, right in zip(phases, phases[1:]))
    assert assist.active is True


def test_trajectory_target_interpolates_normalized_phase_and_pins_reserved():
    start = G20_OPEN_POSE.copy()
    end = start.copy()
    end[1] = 55
    end[11:15] = 0

    target = trajectory_target(np.stack((start, end)), 0.5)

    assert target[1] == 155
    assert np.all(target[list(RESERVED_IDX)] == 255)


def test_trajectory_suffix_starts_at_exact_phase_and_keeps_only_future_frames():
    trajectory = np.stack([
        G20_OPEN_POSE - index * 10 for index in range(5)
    ]).astype(np.float32)
    trajectory[:, list(RESERVED_IDX)] = 0

    suffix = trajectory_suffix(trajectory, 0.375)

    assert suffix.shape == (4, 20)
    assert suffix[0, 0] == 240
    assert suffix[1, 0] == 235
    assert suffix[-1, 0] == 215
    assert np.all(suffix[:, list(RESERVED_IDX)] == 255)


def test_trajectory_suffix_at_endpoint_contains_only_endpoint():
    trajectory = np.stack((G20_OPEN_POSE, G20_OPEN_POSE - 20))

    suffix = trajectory_suffix(trajectory, 1.0)

    assert suffix.shape == (1, 20)
    np.testing.assert_array_equal(
        suffix[0, list(ACTIVE_IDX)], trajectory[-1, list(ACTIVE_IDX)]
    )


def test_nearest_trajectory_suffix_uses_active_pose_and_keeps_future_frames():
    trajectory = np.stack([G20_OPEN_POSE.copy() for _ in range(4)]).astype(
        np.float32
    )
    for frame, value in enumerate((240, 180, 100, 20)):
        trajectory[frame, list(ACTIVE_IDX)] = value
    trajectory[:, list(RESERVED_IDX)] = -500
    current = G20_OPEN_POSE.copy()
    current[list(ACTIVE_IDX)] = 105
    current[list(RESERVED_IDX)] = 999

    suffix, frame_index, error = nearest_trajectory_suffix(trajectory, current)

    assert frame_index == 2
    assert error == 5.0
    assert suffix.shape == (2, 20)
    assert np.all(suffix[0, list(ACTIVE_IDX)] == 100)
    assert np.all(suffix[-1, list(ACTIVE_IDX)] == 20)
    assert np.all(suffix[:, list(RESERVED_IDX)] == 255)


def test_nearest_trajectory_suffix_tie_selects_earliest_frame():
    trajectory = np.stack((G20_OPEN_POSE, G20_OPEN_POSE)).astype(np.float32)

    _suffix, frame_index, error = nearest_trajectory_suffix(
        trajectory, G20_OPEN_POSE
    )

    assert frame_index == 0
    assert error == 0.0


def test_selected_manual_trajectory_can_force_full_action_from_frame_one():
    trajectory = np.stack([G20_OPEN_POSE.copy() for _ in range(4)]).astype(
        np.float32
    )
    for frame, value in enumerate((240, 180, 100, 20)):
        trajectory[frame, list(ACTIVE_IDX)] = value
    trajectory[:, list(RESERVED_IDX)] = -500
    current = G20_OPEN_POSE.copy()
    current[list(ACTIVE_IDX)] = 20

    selected, frame_index, error = selected_manual_trajectory(
        trajectory, current, force_from_start=True
    )

    assert frame_index == 0
    assert error == 220.0
    assert selected.shape == trajectory.shape
    assert np.all(selected[0, list(ACTIVE_IDX)] == 240)
    assert np.all(selected[-1, list(ACTIVE_IDX)] == 20)
    assert np.all(selected[:, list(RESERVED_IDX)] == 255)


def test_selected_manual_trajectory_uses_nearest_suffix_by_default():
    trajectory = np.stack([G20_OPEN_POSE.copy() for _ in range(3)]).astype(
        np.float32
    )
    for frame, value in enumerate((220, 100, 20)):
        trajectory[frame, list(ACTIVE_IDX)] = value
    current = G20_OPEN_POSE.copy()
    current[list(ACTIVE_IDX)] = 20

    selected, frame_index, error = selected_manual_trajectory(
        trajectory, current, force_from_start=False
    )

    assert frame_index == 2
    assert error == 0.0
    assert selected.shape == (1, 20)


def test_manifest_primitive_can_force_number_key_playback_from_frame_one():
    primitive = Primitive(
        6,
        "reverse_one",
        np.stack((G20_OPEN_POSE, G20_OPEN_POSE)),
        (np.zeros((6, 1), dtype=np.float32),),
        manual_from_start=True,
    )

    assert manual_action_starts_from_first_frame(primitive, [])
    assert manual_action_starts_from_first_frame(
        Primitive(
            7,
            "configured",
            primitive.trajectory,
            primitive.templates,
        ),
        [7],
    )
    assert not manual_action_starts_from_first_frame(
        Primitive(
            8,
            "nearest",
            primitive.trajectory,
            primitive.templates,
        ),
        [],
    )


def test_phase_command_is_step_limited_from_observed_state():
    target = G20_OPEN_POSE.copy()
    target[list(ACTIVE_IDX)] = 0
    command = step_limited_command(target, G20_OPEN_POSE, max_step=5)

    deltas = np.abs(
        np.asarray(command)[list(ACTIVE_IDX)]
        - G20_OPEN_POSE[list(ACTIVE_IDX)]
    )
    assert float(np.max(deltas)) <= 5
    assert command[11:15] == [255, 255, 255, 255]


def test_state_guarded_command_advances_from_previous_but_caps_state_lead():
    target = G20_OPEN_POSE.copy()
    target[list(ACTIVE_IDX)] = 0
    observed = G20_OPEN_POSE.copy()

    first = state_guarded_command(
        target, G20_OPEN_POSE, observed, max_step=10, max_state_lead=25
    )
    second = state_guarded_command(
        target, first, observed, max_step=10, max_state_lead=25
    )
    third = state_guarded_command(
        target, second, observed, max_step=10, max_state_lead=25
    )

    assert first[0] == 245
    assert second[0] == 235
    assert third[0] == 230
    assert third[11:15] == [255, 255, 255, 255]


def test_a4_gate_aligns_left_then_closes_tip_then_releases_right_turn():
    close_pose = G20_OPEN_POSE.copy()
    close_pose[[0, 5, 10, 15]] = [254, 0, 51, 0]
    later_target = close_pose.copy()
    later_target[[0, 5, 10, 15]] = [213, 138, 0, 50]
    later_target[1] = 106
    observed = G20_OPEN_POSE.copy()
    observed[[0, 5, 10, 15]] = [230, 70, 30, 40]
    gate = Action4ThumbTipGate(tolerance=5, confirm_frames=3)

    target, waiting, released_now = gate.apply(
        later_target,
        primitive_id=4,
        close_pose=close_pose,
        observed=observed,
        previous=observed,
    )

    assert waiting is True
    assert released_now is False
    np.testing.assert_array_equal(target[[0, 5, 10]], close_pose[[0, 5, 10]])
    assert target[15] == observed[15]
    assert target[1] == later_target[1]
    assert gate.last_applied_stage == "left"

    observed[[0, 5, 10]] = close_pose[[0, 5, 10]]
    for expected_confirmed in (1, 2):
        target, waiting, released_now = gate.apply(
            later_target,
            primitive_id=4,
            close_pose=close_pose,
            observed=observed,
            previous=observed,
        )
        assert waiting is True
        assert released_now is False
        assert gate.confirmed == expected_confirmed
        assert target[15] == 40

    target, waiting, released_now = gate.apply(
        later_target,
        primitive_id=4,
        close_pose=close_pose,
        observed=observed,
        previous=observed,
    )
    assert waiting is True
    assert released_now is False
    assert gate.stage == "tip"
    assert target[15] == 40

    observed[15] = 4
    for expected_confirmed in (1, 2):
        target, waiting, released_now = gate.apply(
            later_target,
            primitive_id=4,
            close_pose=close_pose,
            observed=observed,
            previous=observed,
        )
        assert waiting is True
        assert released_now is False
        assert gate.confirmed == expected_confirmed
        assert target[15] == 0
        np.testing.assert_array_equal(
            target[[0, 5, 10]], close_pose[[0, 5, 10]]
        )

    target, waiting, released_now = gate.apply(
        later_target,
        primitive_id=4,
        close_pose=close_pose,
        observed=observed,
        previous=observed,
    )
    assert waiting is False
    assert released_now is True
    np.testing.assert_array_equal(target[[0, 5, 10]], later_target[[0, 5, 10]])
    assert target[15] == close_pose[15]


def test_a4_gate_resets_when_another_action_is_selected():
    close_pose = G20_OPEN_POSE.copy()
    close_pose[15] = 0
    observed = G20_OPEN_POSE.copy()
    gate = Action4ThumbTipGate(tolerance=5, confirm_frames=3)
    gate.apply(
        close_pose,
        primitive_id=4,
        close_pose=close_pose,
        observed=observed,
        previous=observed,
    )
    assert gate.active is True

    target, waiting, released_now = gate.apply(
        G20_OPEN_POSE,
        primitive_id=2,
        close_pose=close_pose,
        observed=observed,
        previous=observed,
    )

    assert gate.active is False
    assert waiting is False
    assert released_now is False
    np.testing.assert_array_equal(target, G20_OPEN_POSE)


def test_active_pose_error_ignores_reserved_channels():
    observed = G20_OPEN_POSE.copy()
    observed[11:15] = 0
    assert active_pose_error(observed, G20_OPEN_POSE) == 0.0

    observed[15] -= 17
    assert active_pose_error(observed, G20_OPEN_POSE) == 17.0


def test_space_reset_modes_are_explicit_and_have_conservative_defaults():
    default = parse_args(["--library", "library"])
    enabled = parse_args(
        [
            "--library",
            "library",
            "--reset-before-arm",
            "--reset-on-start",
            "--reset-after-disarm",
            "--episode-start-action-end",
            "3",
            "--manual-action-from-start",
            "4",
            "--freeze-thumb-on-start",
            "--startup-thumb-offsets",
            "30,0,-10,5",
            "--minimal-overlay",
        ]
    )

    assert default.reset_before_arm is False
    assert default.reset_on_start is False
    assert default.reset_after_disarm is False
    assert default.episode_start_action_end is None
    assert enabled.reset_before_arm is True
    assert enabled.reset_on_start is True
    assert enabled.reset_after_disarm is True
    assert enabled.episode_start_action_end == 3
    assert default.manual_action_from_start == []
    assert enabled.manual_action_from_start == [4]
    assert default.freeze_thumb_on_start is False
    assert enabled.freeze_thumb_on_start is True
    assert default.startup_thumb_offsets is None
    assert enabled.startup_thumb_offsets == (30.0, 0.0, -10.0, 5.0)
    assert default.startup_thumb_pose is None
    assert default.thumb_roundtrip_key == 6
    assert default.thumb_roundtrip_source_action == 2
    assert default.minimal_overlay is False
    assert enabled.minimal_overlay is True
    assert enabled.reset_tolerance == 12.0
    assert enabled.reset_timeout == 5.0
    assert enabled.reset_confirm_frames == 3
    assert default.a4_thumb_tip_gate is True
    assert default.a4_thumb_tip_tolerance == 5.0
    assert default.a4_thumb_tip_confirm_frames == 3
    assert default.a4_left_align_tolerance == 5.0
    assert default.a4_left_align_confirm_frames == 3
    assert default.a23_spread_routing is True
    assert default.a23_spread_threshold == 0.350
    assert default.a23_spread_hysteresis == 0.030


def test_startup_thumb_offsets_require_freeze_and_startup_reset(tmp_path):
    missing_library = str(tmp_path / "missing")

    assert main([
        "--library",
        missing_library,
        "--startup-thumb-offsets",
        "30,0,0,0",
        "--reset-on-start",
    ]) == 2
    assert main([
        "--library",
        missing_library,
        "--startup-thumb-offsets",
        "30,0,0,0",
        "--freeze-thumb-on-start",
    ]) == 2


def test_absolute_startup_thumb_pose_requires_freeze_and_startup_reset(tmp_path):
    missing_library = str(tmp_path / "missing")

    assert main([
        "--library",
        missing_library,
        "--startup-thumb-pose",
        "116,253,254,118",
        "--reset-on-start",
    ]) == 2
    assert main([
        "--library",
        missing_library,
        "--startup-thumb-pose",
        "116,253,254,118",
        "--freeze-thumb-on-start",
    ]) == 2
    assert main([
        "--library",
        missing_library,
        "--startup-thumb-offsets",
        "30,0,0,0",
        "--startup-thumb-pose",
        "116,253,254,118",
        "--freeze-thumb-on-start",
        "--reset-on-start",
    ]) == 2


def test_post_episode_reset_completes_disarmed_without_starting_episode():
    armed, status = reset_completion(False)

    assert armed is False
    assert status == "RESET COMPLETE; DISARMED at open pose"

    armed, status = reset_completion(False, "action 3 endpoint")
    assert armed is False
    assert status == "RESET COMPLETE; DISARMED at action 3 endpoint"


def test_reset_target_can_preserve_a_frozen_thumb():
    held = G20_OPEN_POSE.copy()
    held[list(THUMB_IDX)] = [70, 80, 90, 30]

    reset_target = frozen_thumb_target(G20_OPEN_POSE, held)

    assert active_pose_error(reset_target, reset_target) == 0.0
    assert active_pose_error(reset_target, G20_OPEN_POSE) > 0.0
    np.testing.assert_array_equal(
        reset_target[list(THUMB_IDX)], held[list(THUMB_IDX)]
    )


def test_delete_last_episode_uses_d_and_r_is_manual_reset():
    assert is_delete_last_episode_key(ord("d"))
    assert is_delete_last_episode_key(ord("D"))
    assert not is_delete_last_episode_key(ord("r"))
    assert not is_delete_last_episode_key(ord("R"))
    assert is_manual_reset_key(ord("r"))
    assert is_manual_reset_key(ord("R"))
    assert not is_manual_reset_key(ord("d"))


def test_t_key_toggles_full_mediapipe_teleop():
    assert is_full_teleop_toggle_key(ord("t"))
    assert is_full_teleop_toggle_key(ord("T"))
    assert not is_full_teleop_toggle_key(ord("0"))


def test_f_key_toggles_thumb_freeze():
    assert is_thumb_freeze_toggle_key(ord("f"))
    assert is_thumb_freeze_toggle_key(ord("F"))
    assert not is_thumb_freeze_toggle_key(ord("t"))


def test_frozen_thumb_target_pins_only_thumb_channels():
    desired = np.arange(20, dtype=np.float32) + 100
    held = np.arange(20, dtype=np.float32) + 20

    target = frozen_thumb_target(desired, held)

    np.testing.assert_array_equal(target[list(THUMB_IDX)], held[list(THUMB_IDX)])
    nonthumb_active = tuple(index for index in ACTIVE_IDX if index not in THUMB_IDX)
    np.testing.assert_array_equal(
        target[list(nonthumb_active)], desired[list(nonthumb_active)]
    )
    assert np.all(target[list(RESERVED_IDX)] == 255)


def test_offset_thumb_pose_changes_only_thumb_and_clips_sdk_range():
    current = G20_OPEN_POSE.copy()
    current[list(THUMB_IDX)] = [5, 100, 250, 20]

    target = offset_thumb_pose(current, [-10, 30, 20, 15])

    np.testing.assert_array_equal(
        target[list(THUMB_IDX)],
        [0, 130, 255, 35],
    )
    nonthumb_active = tuple(index for index in ACTIVE_IDX if index not in THUMB_IDX)
    np.testing.assert_array_equal(
        target[list(nonthumb_active)],
        current[list(nonthumb_active)],
    )
    assert np.all(target[list(RESERVED_IDX)] == 255)


def test_absolute_thumb_pose_changes_only_thumb_and_clips_sdk_range():
    current = G20_OPEN_POSE.copy()
    current[list(THUMB_IDX)] = [5, 100, 250, 20]

    target = absolute_thumb_pose(current, [-10, 253, 300, 118])

    np.testing.assert_array_equal(
        target[list(THUMB_IDX)],
        [0, 253, 255, 118],
    )
    nonthumb_active = tuple(index for index in ACTIVE_IDX if index not in THUMB_IDX)
    np.testing.assert_array_equal(
        target[list(nonthumb_active)],
        current[list(nonthumb_active)],
    )
    assert np.all(target[list(RESERVED_IDX)] == 255)


def test_thumb_roundtrip_uses_nearest_thumb_frame_and_holds_four_fingers():
    source = np.repeat(G20_OPEN_POSE[None, :], 4, axis=0)
    source[:, list(THUMB_IDX)] = np.asarray(
        [
            [240, 230, 220, 210],
            [180, 170, 160, 150],
            [120, 110, 100, 90],
            [60, 50, 40, 30],
        ],
        dtype=np.float32,
    )
    current = G20_OPEN_POSE.copy()
    current[list(THUMB_IDX)] = [175, 165, 155, 145]
    current[[1, 2, 3, 4, 6, 7, 8, 9, 16, 17, 18, 19]] = np.arange(12) + 80

    trajectory, nearest, error = thumb_roundtrip_trajectory(source, current)

    assert nearest == 1
    assert error == 5.0
    expected_thumb = np.concatenate(
        (source[1:, list(THUMB_IDX)], source[-2::-1, list(THUMB_IDX)]),
        axis=0,
    )
    np.testing.assert_array_equal(
        trajectory[:, list(THUMB_IDX)], expected_thumb
    )
    nonthumb_active = tuple(index for index in ACTIVE_IDX if index not in THUMB_IDX)
    np.testing.assert_array_equal(
        trajectory[:, list(nonthumb_active)],
        np.repeat(
            current[None, list(nonthumb_active)],
            len(trajectory),
            axis=0,
        ),
    )
    np.testing.assert_array_equal(
        trajectory[-1, list(THUMB_IDX)], source[0, list(THUMB_IDX)]
    )
    assert np.all(trajectory[:, list(RESERVED_IDX)] == 255)


def _finger_spread_landmarks(spacing: float) -> np.ndarray:
    landmarks = np.zeros((21, 3), dtype=np.float32)
    landmarks[5] = [-0.5, 0.0, 0.0]
    landmarks[17] = [0.5, 0.0, 0.0]
    for offset, tip in enumerate((8, 12, 16, 20)):
        landmarks[tip] = [0.0, offset * spacing, 1.0]
    return landmarks


def test_four_finger_spread_score_is_normalized_by_palm_width():
    landmarks = _finger_spread_landmarks(0.42)

    assert abs(four_finger_spread_score(landmarks) - 0.42) < 1e-6


def test_a23_router_allows_only_a2_when_together():
    router = Action23FingerSpreadRouter(
        threshold=0.350,
        hysteresis=0.030,
    )

    # Values come from the user's current live screenshots: together is
    # 0.240--0.254 and clearly spread is about 0.506.
    assert router.update(_finger_spread_landmarks(0.254)) is False
    assert router.excluded_ids((1, 2, 3, 4, 5)) == {1, 3, 4, 5}
    assert router.allows_a3_assist is False
    assert router.freezes_four_fingers is True
    assert router.update(_finger_spread_landmarks(0.370)) is False
    assert router.update(_finger_spread_landmarks(0.506)) is True
    assert router.excluded_ids((1, 2, 3, 4, 5)) == set()
    assert router.allows_a3_assist is True
    assert router.freezes_four_fingers is False
    assert router.update(_finger_spread_landmarks(0.340)) is True
    assert router.update(_finger_spread_landmarks(0.240)) is False
    assert router.freezes_four_fingers is True


def test_full_mediapipe_map_drives_thumb_fingers_and_spread():
    radians = np.zeros(20, dtype=np.float32)
    radians[[0, 1, 2, 3, 4, 5, 10, 15, 16, 17, 18, 19]] = [
        0.5, 1.0, 1.0, 1.0, 1.0, 0.8, 0.8, 0.7, 1.0, 1.0, 1.0, 1.0
    ]
    radians[[6, 7, 8, 9]] = [0.12, 0.12, -0.12, -0.12]

    target = full_mediapipe_g20_target(radians)

    assert target.shape == (20,)
    assert np.all((target >= 0) & (target <= 255))
    assert np.all(target[list(RESERVED_IDX)] == 255)
    assert np.any(target[[0, 5, 10, 15]] != G20_OPEN_POSE[[0, 5, 10, 15]])
    assert np.all(
        target[[1, 2, 3, 4, 16, 17, 18, 19]]
        < G20_OPEN_POSE[[1, 2, 3, 4, 16, 17, 18, 19]]
    )
    assert np.any(target[[6, 7, 8, 9]] != G20_OPEN_POSE[[6, 7, 8, 9]])


def test_manual_hybrid_keeps_only_action_thumb_channels():
    action = np.arange(20, dtype=np.float32) + 30
    mediapipe = np.arange(20, dtype=np.float32) + 130

    target = action_thumb_mediapipe_fingers_target(action, mediapipe)

    np.testing.assert_array_equal(target[list(THUMB_IDX)], action[list(THUMB_IDX)])
    nonthumb_active = tuple(index for index in ACTIVE_IDX if index not in THUMB_IDX)
    np.testing.assert_array_equal(
        target[list(nonthumb_active)], mediapipe[list(nonthumb_active)]
    )
    assert np.all(target[list(RESERVED_IDX)] == 255)


def test_nonthumb_radian_map_drives_only_four_finger_flexion():
    radians = np.zeros(20, dtype=np.float32)
    radians[[1, 2, 3, 4]] = 1.4
    radians[[16, 17, 18, 19]] = 1.57

    target = nonthumb_radians_to_g20_target(
        radians,
        base_gain=1.0,
        base_gains=(1.0, 1.0, 1.0, 1.0),
        tip_gain=1.0,
        tip_gains=(1.0, 1.0, 1.0, 1.0),
    )

    assert np.all(target[[1, 2, 3, 4, 16, 17, 18, 19]] == 0)
    np.testing.assert_array_equal(target[[0, 5, 10, 15]], G20_OPEN_POSE[[0, 5, 10, 15]])
    np.testing.assert_array_equal(target[[6, 7, 8, 9]], G20_OPEN_POSE[[6, 7, 8, 9]])


def test_hybrid_finger_target_keeps_thumb_and_spread_library_only():
    library = G20_OPEN_POSE.copy()
    library[[0, 5, 10, 15]] = [30, 70, 20, 50]
    library[[6, 7, 8, 9]] = [180, 140, 100, 60]
    mediapipe = np.zeros(20, dtype=np.float32)

    target = hybrid_finger_target(
        library,
        mediapipe,
        base_blend=0.5,
        tip_blend=0.5,
        base_residual_limit=20,
        tip_residual_limit=25,
    )

    np.testing.assert_array_equal(target[[0, 5, 10, 15]], library[[0, 5, 10, 15]])
    np.testing.assert_array_equal(target[[6, 7, 8, 9]], library[[6, 7, 8, 9]])
    np.testing.assert_array_equal(target[11:15], np.full(4, 255))
    np.testing.assert_array_equal(target[[1, 2, 3, 4]], library[[1, 2, 3, 4]] - 20)
    np.testing.assert_array_equal(target[[16, 17, 18, 19]], library[[16, 17, 18, 19]] - 25)


def test_unlocked_fallback_copies_mediapipe_flexion_and_holds_library_channels():
    anchor = G20_OPEN_POSE.copy()
    anchor[[0, 5, 10, 15]] = [30, 70, 20, 50]
    anchor[[6, 7, 8, 9]] = [180, 140, 100, 60]
    mediapipe = np.arange(20, dtype=np.float32) * 7

    target = mediapipe_finger_fallback_target(anchor, mediapipe)

    flexion = [1, 2, 3, 4, 16, 17, 18, 19]
    np.testing.assert_array_equal(target[flexion], mediapipe[flexion])
    np.testing.assert_array_equal(target[[0, 5, 10, 15]], anchor[[0, 5, 10, 15]])
    np.testing.assert_array_equal(target[[6, 7, 8, 9]], anchor[[6, 7, 8, 9]])
    np.testing.assert_array_equal(target[11:15], np.full(4, 255))
