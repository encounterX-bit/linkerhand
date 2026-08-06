from collections import deque
from pathlib import Path

import numpy as np

from scripts.train_g20_visual_act import (
    EpisodeSource,
    MarkerSuccessReference,
    count_thumb_route_reversals,
    marker_layout_error,
    resample_rows_to_fps,
    state_history_vector,
    tactile_vector,
)
from src.comms.visual_act_to_linkerhand import make_observation


def _row(value: int, q5: int = 255, q10: int = 245):
    state = [value] * 20
    action = [255] * 20
    action[5] = q5
    action[10] = q10
    return {"joint_pos": state, "last_action": action}


def test_state_history_vector_is_oldest_to_current_and_clamps_episode_start():
    episode = EpisodeSource(
        Path("episode_000"),
        tuple(_row(value) for value in range(5)),
        "thumb push",
    )

    at_start = state_history_vector(episode, 0, (3, 1, 0))
    assert at_start.shape == (60,)
    assert [int(at_start[i * 20]) for i in range(3)] == [0, 0, 0]

    later = state_history_vector(episode, 4, (3, 1, 0))
    assert [int(later[i * 20]) for i in range(3)] == [1, 3, 4]


def test_thumb_route_filter_counts_q5_and_q10_direction_changes():
    q5 = [255] * 20 + [200] * 20 + [100] * 20 + [200] * 20 + [100] * 20
    q10 = [245] * 40 + [180] * 20 + [100] * 20 + [180] * 20
    rows = tuple(_row(0, side, roll) for side, roll in zip(q5, q10))

    # q5 changes direction twice; q10 changes direction once.
    assert count_thumb_route_reversals(rows) == 3


def test_live_observation_concatenates_exact_state_history():
    frames = deque([np.zeros((8, 8, 3), dtype=np.uint8)], maxlen=1)
    states = deque(([value] * 20 for value in (10, 20, 30)), maxlen=3)

    observation = make_observation(
        frames,
        [30] * 20,
        (0,),
        states,
        (2, 1, 0),
    )

    vector = observation["observation.state"].numpy().reshape(-1)
    assert vector.shape == (60,)
    assert [int(vector[i * 20]) for i in range(3)] == [10, 20, 30]


def test_mass_contact_tactile_layout_matches_training_and_live_inference():
    row = {
        **_row(10),
        "mass_values": [0, 10, 20, 30, 40, 50],
        "contact_6": [0, 0, 1, 1, 1, 1],
    }
    episode = EpisodeSource(Path("episode_000"), (row,), "tactile grasp")

    training = state_history_vector(
        episode, 0, (0,), tactile_mode="mass-contact"
    )
    assert training.shape == (32,)
    np.testing.assert_array_equal(training[-12:], tactile_vector(row, "mass-contact"))

    frames = deque([np.zeros((8, 8, 3), dtype=np.uint8)], maxlen=1)
    live = make_observation(
        frames,
        [10] * 20,
        tactile_mode="mass-contact",
        mass_values=row["mass_values"],
        contact_threshold=20,
    )["observation.state"].numpy().reshape(-1)
    assert live.shape == (32,)
    np.testing.assert_array_equal(live[-12:], training[-12:])


def test_missing_or_invalid_tactile_feedback_is_rejected_for_training():
    assert tactile_vector(_row(0), "mass-contact") is None
    invalid = {
        **_row(0),
        "mass_values": [0, 1, float("nan"), 3, 4, 5],
        "contact_6": [0, 0, 0, 0, 0, 0],
    }
    assert tactile_vector(invalid, "mass-contact") is None


def test_rated_rows_can_be_resampled_from_ten_to_twenty_hz():
    rows = tuple(
        {
            **_row(value),
            "index": index,
            "elapsed": index * 0.1,
            "timestamp": 1000.0 + index * 0.1,
            "image_path": f"images/{index:06d}.jpg",
        }
        for index, value in enumerate((0, 10, 20))
    )

    resampled, observed_fps = resample_rows_to_fps(rows, 20)

    assert observed_fps == 10
    assert len(resampled) == 5
    assert [round(row["elapsed"], 2) for row in resampled] == [
        0.0,
        0.05,
        0.1,
        0.15,
        0.2,
    ]
    assert [round(row["joint_pos"][0]) for row in resampled] == [0, 5, 10, 15, 20]
    assert [round(row["last_action"][5]) for row in resampled] == [
        255,
        255,
        255,
        255,
        255,
    ]


def test_success_marker_layout_ignores_translation_and_scale_but_not_rotation():
    centers = {
        20: np.asarray([0.0, 0.0]),
        21: np.asarray([2.0, 0.0]),
        22: np.asarray([0.0, 1.0]),
        23: np.asarray([2.0, 1.0]),
    }
    reference = MarkerSuccessReference(
        Path("success.jpg"), "DICT_4X4_50", centers
    )
    translated_scaled = {
        marker_id: point * 3.5 + np.asarray([120.0, -40.0])
        for marker_id, point in centers.items()
    }
    rotated = {
        marker_id: np.asarray([-point[1], point[0]])
        for marker_id, point in centers.items()
    }

    invariant_error, missing = marker_layout_error(reference, translated_scaled)
    rotation_error, _ = marker_layout_error(reference, rotated)

    assert missing == ()
    assert invariant_error is not None and invariant_error < 1e-6
    assert rotation_error is not None and rotation_error > 0.5


def test_success_marker_layout_reports_missing_reference_ids():
    reference = MarkerSuccessReference(
        Path("success.jpg"),
        "DICT_4X4_50",
        {
            20: np.asarray([0.0, 0.0]),
            21: np.asarray([1.0, 0.0]),
            22: np.asarray([0.0, 1.0]),
        },
    )

    error, missing = marker_layout_error(
        reference,
        {20: np.asarray([0.0, 0.0]), 22: np.asarray([0.0, 1.0])},
    )

    assert error is None
    assert missing == (21,)
