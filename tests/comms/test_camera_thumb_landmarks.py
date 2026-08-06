from types import SimpleNamespace

import numpy as np

from src.comms.camera_to_linkerhand import (
    FreshHandGate,
    LinkerHandHardwareSink,
    MotionGate,
    Q0KeyTrim,
    ThumbKeyboardControl,
    _handle_thumb_key,
    _hardware_candidate_adjuster,
    apply_q0_live_trim,
    apply_thumb_keyboard_control,
    manual_thumb_target,
)


def test_stopped_hold_gate_publishes_no_position_command():
    class FakeNode:
        def __init__(self):
            self.sessions = []
            self.poses = []

        def publish_session_active(self, active):
            self.sessions.append(bool(active))

        def publish_pose(self, pose):
            self.poses.append(list(pose))

    sink = object.__new__(LinkerHandHardwareSink)
    sink.node = FakeNode()
    sink.motion_gate = MotionGate(active=False, stop_mode="hold")
    sink.thumb_keys = None

    sink.set_joints(np.zeros(20, dtype=float))
    sink.idle_tick()

    assert sink.node.sessions == [False]
    assert sink.node.poses == []


def _candidate():
    return {"side": "right", "joint_rad": [0.0] * 20}


def _landmarks(*, thumb_line_distance: float) -> np.ndarray:
    """Minimal hand with MCP roots on x and a thumb reaching across them."""
    lm = np.zeros((21, 3), dtype=float)
    lm[5] = (0.0, 0.0, 0.0)
    lm[9] = (1.0, 0.0, 0.0)
    lm[13] = (2.0, 0.0, 0.0)
    lm[17] = (3.0, 0.0, 0.0)
    lm[2] = (0.3, thumb_line_distance, 0.0)
    lm[3] = (1.2, thumb_line_distance, 0.0)
    lm[4] = (2.1, thumb_line_distance, 0.0)
    return lm


def _adjuster():
    return _hardware_candidate_adjuster(
        1.0,
        landmark_thumb=True,
        landmark_thumb_gain=1.0,
        landmark_thumb_reach_gain=1.0,
        landmark_spread=False,
        landmark_spread_gain=0.0,
        landmark_spread_limit=0.0,
        landmark_spread_calibration_frames=1,
    )


def test_thumb_landmarks_drive_base_downpress_and_side_swing_near_palm():
    out = _adjuster()(_candidate(), landmarks=_landmarks(thumb_line_distance=0.0))

    assert out["joint_rad"][0] > 0.0
    assert out["joint_rad"][5] > 0.0
    assert out["joint_rad"][10] == 0.0


def test_open_thumb_does_not_receive_fixed_base_pose():
    out = _adjuster()(_candidate(), landmarks=_landmarks(thumb_line_distance=3.0))

    assert out["joint_rad"][0] == 0.0
    assert out["joint_rad"][5] == 0.0


def test_hardware_fresh_hand_gate_never_passes_held_landmarks():
    gate = FreshHandGate(confirm_frames=3)
    fresh = SimpleNamespace(detected=True, held=False)
    held = SimpleNamespace(detected=False, held=True)

    assert not gate.update(fresh)
    assert not gate.update(fresh)
    assert gate.update(fresh)
    assert gate.tracking

    assert not gate.update(held)
    assert not gate.tracking
    assert gate.streak == 0

    assert not gate.update(fresh)
    assert not gate.update(None)
    assert gate.streak == 0


def test_q0_key_trim_changes_only_command_index_0():
    trim = Q0KeyTrim(step=5)
    pose = [100] * 20

    trim.adjust(+1)
    assert apply_q0_live_trim(pose, trim) == [105] + pose[1:]

    trim.offset = -200
    assert apply_q0_live_trim(pose, trim)[0] == 0


def test_q0_key_trim_has_no_saturation_windup():
    trim = Q0KeyTrim(step=5)
    pose = [100] * 20

    trim.offset = 255
    assert apply_q0_live_trim(pose, trim)[0] == 255
    assert trim.offset == 155

    trim.adjust(-1)
    assert trim.offset == 150
    assert apply_q0_live_trim(pose, trim)[0] == 250


def test_q0_keyboard_uses_w_and_s_not_equals_and_minus():
    keys = ThumbKeyboardControl(q0_step=5, tip_step=10)

    assert _handle_thumb_key(ord("w"), keys)
    assert keys.q0.offset == 5
    assert _handle_thumb_key(ord("S"), keys)
    assert keys.q0.offset == 0
    assert not _handle_thumb_key(ord("="), keys)
    assert not _handle_thumb_key(ord("-"), keys)


def test_jl_and_ik_control_only_the_two_added_thumb_channels():
    keys = ThumbKeyboardControl(q0_step=5, abd_step=7, roll_step=9, tip_step=10)
    pose = [100] * 20
    open_range = [255] * 20

    assert _handle_thumb_key(ord("L"), keys)
    out = apply_thumb_keyboard_control(pose, keys, open_range)
    assert [i for i, (a, b) in enumerate(zip(pose, out)) if a != b] == [5]
    assert out[5] == 107

    assert _handle_thumb_key(ord("j"), keys)
    assert _handle_thumb_key(ord("i"), keys)
    out = apply_thumb_keyboard_control(pose, keys, open_range)
    assert [i for i, (a, b) in enumerate(zip(pose, out)) if a != b] == [10]
    assert out[10] == 109

    assert _handle_thumb_key(ord("K"), keys)
    assert apply_thumb_keyboard_control(pose, keys, open_range) == pose


def test_ad_controls_only_thumb_tip_and_r_holds_thumb_open():
    keys = ThumbKeyboardControl(q0_step=5, tip_step=10)
    pose = [100] * 20
    open_range = [255] * 20
    open_range[10] = 245

    assert _handle_thumb_key(ord("d"), keys)
    out = apply_thumb_keyboard_control(pose, keys, open_range)
    assert [i for i, (a, b) in enumerate(zip(pose, out)) if a != b] == [15]
    assert out[15] == 110

    assert _handle_thumb_key(ord("w"), keys)
    assert _handle_thumb_key(ord("l"), keys)
    assert _handle_thumb_key(ord("i"), keys)
    assert _handle_thumb_key(ord("r"), keys)
    assert [keys.q0.offset, keys.abd.offset, keys.roll.offset, keys.tip.offset] == [0, 0, 0, 0]
    out = apply_thumb_keyboard_control(pose, keys, open_range)
    assert [out[i] for i in (0, 5, 10, 15)] == [255, 255, 245, 255]

    assert _handle_thumb_key(ord("a"), keys)
    assert not keys.return_to_open


def test_manual_thumb_target_works_without_camera_and_does_not_drift():
    keys = ThumbKeyboardControl(q0_step=10, abd_step=10, roll_step=10, tip_step=10)
    open_range = [255] * 20
    open_range[10] = 245
    keys.seed_from_pose(open_range)

    assert _handle_thumb_key(ord("j"), keys)
    assert keys.manual_override
    first = manual_thumb_target(open_range, keys, open_range)
    second = manual_thumb_target(first, keys, open_range)

    assert first[5] == 245
    assert second == first
    assert [i for i, (a, b) in enumerate(zip(open_range, first)) if a != b] == [5]


def test_r_returns_manual_thumb_target_to_seeded_g20_open_pose():
    keys = ThumbKeyboardControl(q0_step=10, abd_step=10, roll_step=10, tip_step=10)
    open_range = [255] * 20
    open_range[10] = 245
    keys.seed_from_pose(open_range)
    for key in "sjka":
        assert _handle_thumb_key(ord(key), keys)

    moved = manual_thumb_target(open_range, keys, open_range)
    assert [moved[i] for i in (0, 5, 10, 15)] == [245, 245, 235, 245]

    assert _handle_thumb_key(ord("r"), keys)
    returned = manual_thumb_target(moved, keys, open_range)
    assert [returned[i] for i in (0, 5, 10, 15)] == [255, 255, 245, 255]


def test_q_toggles_mediapipe_thumb_and_holds_only_thumb_channels():
    keys = ThumbKeyboardControl(q0_step=10, abd_step=10, roll_step=10, tip_step=10)
    current = [100] * 20
    keys.seed_from_pose(current)

    assert _handle_thumb_key(ord("q"), keys)
    assert not keys.mediapipe_thumb_enabled
    assert keys.manual_override

    camera_pose = [50] * 20
    held = manual_thumb_target(camera_pose, keys, [255] * 20)
    assert [held[i] for i in (0, 5, 10, 15)] == [100, 100, 100, 100]
    assert all(held[i] == 50 for i in range(20) if i not in (0, 5, 10, 15))

    assert _handle_thumb_key(ord("q"), keys)
    assert keys.mediapipe_thumb_enabled
    assert not keys.manual_override


def test_hardware_manual_thumb_tick_is_step_limited_without_mediapipe():
    class FakeNode:
        def __init__(self, state):
            self.last_state = list(state)
            self.published = []

        def publish_pose(self, pose):
            self.published.append(list(pose))

    open_range = [255] * 20
    open_range[10] = 245
    keys = ThumbKeyboardControl(q0_step=10, abd_step=10, roll_step=10, tip_step=10)
    keys.seed_from_pose(open_range)
    node = FakeNode(open_range)
    sink = LinkerHandHardwareSink(
        node,
        side="right", enable_motion=True, max_range_step=5, log_period=999.0,
        relative_mode=False, calibration_frames=1, relative_scale=1.0,
        max_relative_delta=0, open_range=open_range, hardware_map="g20-sim",
        roll_range_ticks=100.0, base_gain=1.0,
        base_gains=(1.0, 1.0, 1.0, 1.0), spread_gain=1.0, tip_gain=1.0,
        tip_gains=(1.0, 1.0, 1.0, 1.0),
        spread_signs=(1.0, 1.0, 1.0, 1.0),
        thumb_base_gain=1.0, thumb_abd_gain=1.0,
        thumb_roll_gain=1.0, thumb_tip_gain=1.0,
        thumb_base_offset=0, thumb_abd_offset=0,
        thumb_roll_offset=0, thumb_tip_offset=0,
        nonthumb_close_deadzone=0, collision_guard=False,
        thumb_safe_mode="free", max_thumb_delta=255,
        max_thumb_abd_delta=255, max_thumb_base_delta=None,
        max_spread_delta=255, spread_close_threshold=1.0,
        spread_recenter_gain=0.0, min_spread_gap=0,
        thumb_index_guard=False, thumb_index_threshold=1.0,
        thumb_index_release=0, thumb_keys=keys,
    )
    sink.prev_range = list(open_range)

    assert _handle_thumb_key(ord("j"), keys)
    assert sink.publish_manual_thumb_tick()
    assert node.published[-1][5] == 250
    assert sink.publish_manual_thumb_tick()
    assert node.published[-1][5] == 245
    assert node.published[-1][1:] == open_range[1:5] + [245] + open_range[6:]
