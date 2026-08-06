from src.comms.visual_act_to_linkerhand import (
    CubeMarkerPose,
    cube_pose_matches,
    finger_mass_contact_count,
    load_cube_ready_profile,
    save_cube_ready_profile,
)


def _pose(**changes):
    values = {
        "center_x": 0.4,
        "center_y": 0.5,
        "width": 0.12,
        "height": 0.20,
        "marker_count": 4,
        "face_id": 2,
    }
    values.update(changes)
    return CubeMarkerPose(**values)


def test_cube_ready_matches_nearby_pose_and_ignores_visible_face_id():
    matched, center_error, scale_error = cube_pose_matches(
        _pose(center_x=0.43, face_id=5),
        _pose(),
        center_tolerance=0.06,
        scale_tolerance=0.30,
    )

    assert matched
    assert center_error < 0.06
    assert scale_error == 0.0


def test_cube_ready_rejects_wrong_position_or_scale():
    wrong_center = cube_pose_matches(
        _pose(center_x=0.52),
        _pose(),
        center_tolerance=0.06,
        scale_tolerance=0.30,
    )[0]
    wrong_scale = cube_pose_matches(
        _pose(width=0.06),
        _pose(),
        center_tolerance=0.06,
        scale_tolerance=0.30,
    )[0]

    assert not wrong_center
    assert not wrong_scale


def test_cube_ready_can_ignore_scale_but_still_checks_position():
    matched = cube_pose_matches(
        _pose(width=0.04, height=0.50),
        _pose(),
        center_tolerance=0.06,
        scale_tolerance=None,
    )[0]
    wrong_center = cube_pose_matches(
        _pose(center_x=0.52, width=0.04, height=0.50),
        _pose(),
        center_tolerance=0.06,
        scale_tolerance=None,
    )[0]

    assert matched
    assert not wrong_center


def test_cube_ready_profile_round_trip(tmp_path):
    path = tmp_path / "camera0.json"
    pose = _pose()

    save_cube_ready_profile(path, pose)

    assert load_cube_ready_profile(path) == pose


def test_finger_mass_contact_gate_uses_fresh_touch_and_threshold():
    sample = {
        "touch_fresh": True,
        "mass_values": [0.0, 4.9, 5.0, 12.0, 0.0, 100.0],
    }

    assert finger_mass_contact_count(sample, 5.0) == 2
    sample["touch_fresh"] = False
    assert finger_mass_contact_count(sample, 5.0) == 0
