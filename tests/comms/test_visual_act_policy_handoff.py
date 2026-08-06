import argparse

import pytest

from src.comms.visual_act_to_linkerhand import (
    PolicyHandoffController,
    parse_g20_pose,
)


def _pose(value: int = 100) -> list[int]:
    pose = [value] * 20
    for index in (11, 12, 13, 14):
        pose[index] = 255
    return pose


def test_parse_g20_pose_requires_twenty_bounded_values() -> None:
    parsed = parse_g20_pose(",".join(str(value) for value in range(20)))
    assert parsed == tuple(float(value) for value in range(20))

    with pytest.raises(argparse.ArgumentTypeError):
        parse_g20_pose("1,2,3")
    with pytest.raises(argparse.ArgumentTypeError):
        parse_g20_pose(",".join(["300"] * 20))


def test_handoff_moves_confirms_warms_and_completes() -> None:
    target = tuple(float(value) for value in _pose(100))
    controller = PolicyHandoffController(
        target_pose=target,
        tolerance=4,
        confirm_frames=3,
        warmup_seconds=2,
        timeout_seconds=10,
    )
    controller.begin(now=0)

    assert controller.update(_pose(80), now=0.1) == "moving"
    assert controller.update(_pose(98), now=0.2) == "moving"
    assert controller.update(_pose(99), now=0.3) == "moving"
    assert controller.update(_pose(100), now=0.4) == "warmup_started"
    assert controller.update(_pose(100), now=1.4) == "warming"
    assert controller.update(_pose(100), now=2.4) == "complete"
    assert not controller.active


def test_handoff_command_is_step_limited_and_timeout_is_safe() -> None:
    controller = PolicyHandoffController(
        target_pose=tuple(float(value) for value in _pose(100)),
        tolerance=2,
        confirm_frames=2,
        warmup_seconds=1,
        timeout_seconds=2,
    )
    controller.begin(now=0)

    command = controller.command(_pose(50), step=10)
    assert command[0] == 60
    assert [command[index] for index in (11, 12, 13, 14)] == [255] * 4
    assert controller.update(_pose(50), now=2.1) == "timeout"
    assert not controller.active


def test_handoff_restarts_warmup_if_pose_drifts() -> None:
    controller = PolicyHandoffController(
        target_pose=tuple(float(value) for value in _pose(100)),
        tolerance=3,
        confirm_frames=1,
        warmup_seconds=2,
        timeout_seconds=10,
    )
    controller.begin(now=0)

    assert controller.update(_pose(100), now=0.1) == "warmup_started"
    assert controller.update(_pose(110), now=1.0) == "warmup_reset"
    assert controller.update(_pose(100), now=2.5) == "warming"
    assert controller.update(_pose(100), now=3.1) == "complete"
