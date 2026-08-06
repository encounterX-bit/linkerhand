import json

import numpy as np

from src.comms.visual_act_to_linkerhand import (
    DemoEndpointStopper,
    load_demo_endpoint_profile,
    observed_hold_pose,
)


ACTIVE = tuple(index for index in range(20) if index not in (11, 12, 13, 14))


def _pose(value=100):
    return [value] * 20


def test_endpoint_stopper_requires_departure_time_and_confirmation():
    endpoint = np.asarray([_pose(80)], dtype=np.float32)
    stopper = DemoEndpointStopper(
        endpoint,
        ACTIVE,
        tolerance=5,
        confirm_frames=3,
        min_active_seconds=2,
        departure_delta=20,
    )
    stopper.reset(_pose(80), now=0)

    assert not stopper.update(_pose(80), now=3)
    assert not stopper.departed

    assert not stopper.update(_pose(120), now=3.1)
    assert stopper.departed
    assert not stopper.update(_pose(82), now=3.2)
    assert not stopper.update(_pose(81), now=3.3)
    assert stopper.update(_pose(80), now=3.4)


def test_endpoint_stopper_does_not_call_an_intermediate_stall_done():
    stopper = DemoEndpointStopper(
        np.asarray([_pose(80)], dtype=np.float32),
        ACTIVE,
        tolerance=5,
        confirm_frames=2,
        min_active_seconds=0,
        departure_delta=20,
    )
    stopper.reset(_pose(120), now=0)

    assert not stopper.update(_pose(100), now=1)
    assert not stopper.update(_pose(100), now=2)
    assert stopper.nearest_error == 20
    assert stopper.confirmed == 0


def test_endpoint_profile_load_and_observed_hold(tmp_path):
    path = tmp_path / "endpoints.json"
    path.write_text(
        json.dumps(
            {
                "schema": "g20_demo_endpoint_profile_v1",
                "active_indices": list(ACTIVE),
                "templates": [{"position": _pose(42)}],
            }
        ),
        encoding="utf-8",
    )

    active, templates = load_demo_endpoint_profile(path)

    assert active == ACTIVE
    assert templates.shape == (1, 20)
    state = list(range(20))
    hold = observed_hold_pose(state)
    assert [hold[index] for index in (11, 12, 13, 14)] == [255] * 4
    assert hold[:11] == state[:11]
