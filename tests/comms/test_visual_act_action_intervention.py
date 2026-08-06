from types import SimpleNamespace

import numpy as np

from src.comms.visual_act_to_linkerhand import (
    ACTIVE_IDX,
    RESERVED_IDX,
    ActionLibraryIntervention,
    apply_thumb_joint_bias,
)


def _pose(value: float) -> np.ndarray:
    pose = np.full(20, value, dtype=np.float32)
    pose[list(RESERVED_IDX)] = 255.0
    return pose


def _controller(trajectory: np.ndarray) -> ActionLibraryIntervention:
    primitive = SimpleNamespace(
        id=2,
        name="test action",
        trajectory=np.asarray(trajectory, dtype=np.float32),
    )
    library = SimpleNamespace(primitives={2: primitive})
    return ActionLibraryIntervention(library, max_step=10, blend_frames=2)


def test_intervention_selects_nearest_active_joint_frame() -> None:
    trajectory = np.stack([_pose(20), _pose(80), _pose(140)])
    observed = _pose(78)
    # Reserved slots must not affect nearest-frame matching.
    observed[list(RESERVED_IDX)] = 0
    controller = _controller(trajectory)

    nearest, error, _frames = controller.begin(2, observed.tolist())

    assert nearest == 1
    assert np.isclose(error, 2.0)
    assert controller.active


def test_intervention_blends_safely_reaches_endpoint_and_holds() -> None:
    trajectory = np.stack([_pose(30), _pose(60), _pose(95)])
    observed = _pose(0)
    controller = _controller(trajectory)
    _, _, frame_count = controller.begin(2, observed.tolist())

    outputs = [controller.next_target() for _ in range(frame_count)]
    active = list(ACTIVE_IDX)
    previous = observed
    for output in outputs:
        delta = np.abs(np.asarray(output)[active] - previous[active])
        assert float(delta.max()) <= 10.0
        assert [output[index] for index in RESERVED_IDX] == [255] * 4
        previous = np.asarray(output)

    assert np.allclose(np.asarray(outputs[-1])[active], 95)
    assert controller.holding
    assert controller.next_target() == outputs[-1]


def test_thumb_biases_remain_available_during_library_hold() -> None:
    trajectory = np.stack([_pose(50), _pose(60)])
    controller = _controller(trajectory)
    controller.begin(2, _pose(50).tolist())
    target = controller.next_target()
    while not controller.holding:
        target = controller.next_target()

    target = apply_thumb_joint_bias(target, 15, -20)
    target = apply_thumb_joint_bias(target, 5, 15)

    assert target[15] == 40
    assert target[5] == 75
    assert all(
        target[index] == 60
        for index in ACTIVE_IDX
        if index not in (5, 15)
    )

    controller.stop()
    assert not controller.active
