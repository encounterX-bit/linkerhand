import numpy as np

from src.comms.visual_act_to_linkerhand import (
    ChunkBoundaryBlender,
    ThumbTipBeforeTurnGate,
    adjust_bounded_keyboard_bias,
    apply_thumb_final_push_offset,
    apply_thumb_joint_bias,
)


def _pose(q0=254, q5=0, q10=51, q15=40):
    pose = [255] * 20
    pose[0] = q0
    pose[5] = q5
    pose[10] = q10
    pose[15] = q15
    return pose


def test_thumb_gate_closes_tip_before_releasing_turn():
    gate = ThumbTipBeforeTurnGate(
        release_threshold=8,
        confirm_frames=3,
        enabled=True,
    )
    turning_target = _pose(q0=220, q5=80, q10=20, q15=0)

    output, waiting, released = gate.apply(turning_target, _pose(q15=35))
    assert waiting and not released
    assert [output[i] for i in (0, 5, 10, 15)] == [254, 0, 51, 0]

    for tip in (7, 6):
        output, waiting, released = gate.apply(turning_target, _pose(q15=tip))
        assert waiting and not released
        assert [output[i] for i in (0, 5, 10, 15)] == [254, 0, 51, 0]

    output, waiting, released = gate.apply(turning_target, _pose(q15=5))
    assert waiting and not released
    assert [output[i] for i in (0, 5, 10, 15)] == [220, 80, 20, 0]

    # During the turn q15 remains closed even if ACT asks to reopen it.
    reopening_target = _pose(q0=200, q5=100, q10=0, q15=120)
    for q0 in (230, 219):
        output, waiting, released = gate.apply(
            reopening_target,
            _pose(q0=q0, q5=70, q10=5, q15=0),
        )
        assert waiting and not released
        assert output[15] == 0

    output, waiting, released = gate.apply(
        reopening_target,
        _pose(q0=218, q5=75, q10=4, q15=0),
    )
    assert waiting and not released
    assert output[15] == 0

    output, waiting, released = gate.apply(
        reopening_target,
        _pose(q0=217, q5=80, q10=3, q15=0),
    )
    assert not waiting and released
    assert [output[i] for i in (0, 5, 10, 15)] == [200, 100, 0, 0]

    output, waiting, released = gate.apply(
        reopening_target,
        _pose(q0=210, q5=90, q10=0, q15=0),
    )
    assert not waiting and not released
    assert output == reopening_target


def test_thumb_gate_does_not_activate_away_from_a4_preturn_pose():
    gate = ThumbTipBeforeTurnGate(
        release_threshold=8,
        confirm_frames=3,
        enabled=True,
    )
    target = _pose(q0=180, q5=90, q10=10, q15=0)
    output, waiting, released = gate.apply(
        target,
        _pose(q0=120, q5=200, q10=180, q15=100),
    )

    assert output == target
    assert not waiting
    assert not released


def test_thumb_final_push_adds_ten_only_after_a4_alignment():
    target = _pose(q0=83, q5=130, q10=35, q15=50)
    observed = _pose(q0=120, q5=125, q10=40, q15=45)

    output, active = apply_thumb_final_push_offset(target, observed, 10)

    assert active
    assert output[15] == 60
    assert output[:15] == target[:15]


def test_thumb_final_push_does_not_change_a2_or_unaligned_a4():
    action2 = _pose(q0=255, q5=60, q10=255, q15=25)
    output, active = apply_thumb_final_push_offset(action2, action2, 10)
    assert not active
    assert output == action2

    final_a4 = _pose(q0=83, q5=130, q10=35, q15=50)
    unaligned = _pose(q0=180, q5=80, q10=80, q15=20)
    output, active = apply_thumb_final_push_offset(final_a4, unaligned, 10)
    assert not active
    assert output == final_a4


def test_thumb_joint_bias_changes_only_selected_channel_and_clamps():
    target = _pose(q0=120, q5=130, q10=140, q15=20)

    output = apply_thumb_joint_bias(target, 15, -10)
    assert output[15] == 10
    assert output[:15] == target[:15]
    assert output[16:] == target[16:]

    assert apply_thumb_joint_bias(target, 15, -50)[15] == 0
    assert apply_thumb_joint_bias(target, 0, 200)[0] == 255


def test_tip_and_side_biases_change_only_q15_and_q5():
    target = _pose(q0=120, q5=130, q10=140, q15=20)

    output = apply_thumb_joint_bias(target, 15, -10)
    output = apply_thumb_joint_bias(output, 5, 25)

    assert output[5] == 155
    assert output[15] == 10
    assert all(
        output[index] == target[index]
        for index in range(20)
        if index not in (5, 15)
    )


def test_bounded_keyboard_bias_clips_both_directions():
    assert adjust_bounded_keyboard_bias(75, 10, 80) == 80
    assert adjust_bounded_keyboard_bias(-75, -10, 80) == -80
    assert adjust_bounded_keyboard_bias(15, -5, 80) == 10


def test_chunk_boundary_blender_only_crossfades_new_chunk_prefix():
    blender = ChunkBoundaryBlender(action_horizon=4, blend_frames=2)
    previous = None

    for value in (0.0, 10.0, 20.0, 30.0):
        raw = np.full(20, value, dtype=np.float32)
        output, active, weight, boundary = blender.apply(raw, previous)
        assert np.allclose(output, raw)
        assert not active
        assert weight == 1.0
        assert not boundary
        previous = output

    output, active, weight, boundary = blender.apply(
        np.full(20, 100.0, dtype=np.float32),
        previous,
    )
    assert boundary and active
    assert weight == 0.5
    assert np.allclose(output, 65.0)

    output, active, weight, boundary = blender.apply(
        np.full(20, 120.0, dtype=np.float32),
        output,
    )
    assert not boundary and active
    assert weight == 1.0
    assert np.allclose(output, 120.0)


def test_chunk_boundary_blender_reset_restarts_without_crossfade():
    blender = ChunkBoundaryBlender(action_horizon=2, blend_frames=1)
    previous = np.zeros(20, dtype=np.float32)
    for value in (10.0, 20.0):
        previous, *_ = blender.apply(
            np.full(20, value, dtype=np.float32),
            previous,
        )

    blender.reset()
    output, active, _weight, boundary = blender.apply(
        np.full(20, 100.0, dtype=np.float32),
        previous,
    )
    assert np.allclose(output, 100.0)
    assert not active
    assert not boundary


def test_chunk_boundary_blender_can_limit_crossfade_to_thumb():
    thumb_indices = (0, 5, 10, 15)
    blender = ChunkBoundaryBlender(
        action_horizon=2,
        blend_frames=2,
        blend_indices=thumb_indices,
    )
    previous = np.full(20, 20.0, dtype=np.float32)
    blender.apply(np.full(20, 10.0, dtype=np.float32), previous)
    blender.apply(np.full(20, 20.0, dtype=np.float32), previous)

    raw = np.full(20, 100.0, dtype=np.float32)
    output, active, weight, boundary = blender.apply(raw, previous)

    assert boundary and active
    assert weight == 0.5
    assert np.allclose(output[list(thumb_indices)], 60.0)
    finger_indices = [i for i in range(20) if i not in thumb_indices]
    assert np.allclose(output[finger_indices], raw[finger_indices])
