import json
from pathlib import Path

import numpy as np

from src.comms.action_library import ActionLibrary
from src.comms.tune_action_spread_pinch import tune_action_spread_pinch


def _library(root: Path) -> tuple[np.ndarray, np.ndarray]:
    folder = root / "primitive_006_reverse"
    folder.mkdir(parents=True)
    trajectory = np.stack(
        (
            np.arange(20, dtype=np.float32) + 200,
            np.arange(20, dtype=np.float32) + 180,
            np.arange(20, dtype=np.float32) + 160,
        )
    )
    trajectory[:, 11:15] = 255
    hand = np.arange(21 * 3, dtype=np.float32).reshape(21, 3) / 100.0
    template = np.repeat(hand[None, :, :], 6, axis=0)
    np.save(folder / "robot_trajectory.npy", trajectory, allow_pickle=False)
    np.save(folder / "human_take_000.npy", template, allow_pickle=False)
    np.save(folder / "human_take_001.npy", template, allow_pickle=False)
    manifest = {
        "schema": ActionLibrary.SCHEMA,
        "fps": 30.0,
        "primitives": [
            {
                "id": 6,
                "name": "reverse",
                "robot_trajectory": (
                    "primitive_006_reverse/robot_trajectory.npy"
                ),
                "human_templates": [
                    "primitive_006_reverse/human_take_000.npy",
                    "primitive_006_reverse/human_take_001.npy",
                ],
                "manual_from_start": True,
            }
        ],
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    return trajectory, template


def test_tune_action_spread_pinch_adds_hold_pinch_then_original_tail(tmp_path):
    original, template = _library(tmp_path)
    target = np.asarray([125, 129, 125, 130], dtype=np.float32)

    result = tune_action_spread_pinch(
        tmp_path,
        primitive_id=6,
        spread_target=target,
        transition_frames=4,
        archive_name="before",
    )

    library = ActionLibrary.load(tmp_path)
    primitive = library.primitives[6]
    tuned = primitive.trajectory
    assert primitive.best_effort_spread_feedback is True
    assert tuned.shape == (7, 20)
    np.testing.assert_array_equal(tuned[0], original[0])
    np.testing.assert_array_equal(tuned[4, [6, 7, 8, 9]], target)
    np.testing.assert_array_equal(
        tuned[:5, [0, 1, 2, 3, 4, 5, 10, 15, 16, 17, 18, 19]],
        np.repeat(
            original[0, [0, 1, 2, 3, 4, 5, 10, 15, 16, 17, 18, 19]][
                None, :
            ],
            5,
            axis=0,
        ),
    )
    np.testing.assert_array_equal(
        tuned[5:, [0, 1, 2, 3, 4, 5, 10, 15, 16, 17, 18, 19]],
        original[1:, [0, 1, 2, 3, 4, 5, 10, 15, 16, 17, 18, 19]],
    )
    np.testing.assert_array_equal(tuned[5:, [6, 7, 8, 9]], [target, target])
    np.testing.assert_array_equal(
        np.load(
            tmp_path / "archive" / "before" / "robot_trajectory.npy",
            allow_pickle=False,
        ),
        original,
    )
    np.testing.assert_array_equal(
        np.load(
            tmp_path
            / "primitive_006_reverse"
            / "human_take_000.npy",
            allow_pickle=False,
        ),
        template,
    )
    assert result["original_frames"] == 3
    assert result["new_frames"] == 7


def test_retune_replaces_existing_prelude_instead_of_stacking_it(tmp_path):
    original, _ = _library(tmp_path)
    first_target = np.asarray([125, 129, 125, 130], dtype=np.float32)
    tighter_target = np.asarray([115, 129, 125, 140], dtype=np.float32)
    tune_action_spread_pinch(
        tmp_path,
        primitive_id=6,
        spread_target=first_target,
        transition_frames=4,
        archive_name="before_first",
    )

    result = tune_action_spread_pinch(
        tmp_path,
        primitive_id=6,
        spread_target=tighter_target,
        transition_frames=4,
        archive_name="before_tighter",
    )

    primitive = ActionLibrary.load(tmp_path).primitives[6]
    tuned = primitive.trajectory
    assert result["original_frames"] == 7
    assert result["new_frames"] == 7
    assert tuned.shape == (7, 20)
    np.testing.assert_array_equal(tuned[0], original[0])
    np.testing.assert_array_equal(
        tuned[4:, [6, 7, 8, 9]],
        [tighter_target, tighter_target, tighter_target],
    )
    np.testing.assert_array_equal(
        tuned[5:, [0, 1, 2, 3, 4, 5, 10, 15, 16, 17, 18, 19]],
        original[1:, [0, 1, 2, 3, 4, 5, 10, 15, 16, 17, 18, 19]],
    )
    manifest = json.loads(
        (tmp_path / "manifest.json").read_text(encoding="utf-8")
    )
    override = manifest["primitives"][0]["spread_pinch_override"]
    assert override["original_frames"] == 3
    assert override["new_frames"] == 7
    assert override["previous_target_q6_q7_q8_q9"] == first_target.tolist()


def test_outer_finger_hold_caps_only_tail_index_and_little_flexion(tmp_path):
    original, _ = _library(tmp_path)
    target = np.asarray([115, 129, 125, 140], dtype=np.float32)
    cap = 185.0

    tune_action_spread_pinch(
        tmp_path,
        primitive_id=6,
        spread_target=target,
        transition_frames=4,
        outer_flexion_max=cap,
        archive_name="before_outer_hold",
    )

    tuned = ActionLibrary.load(tmp_path).primitives[6].trajectory
    # The full pinch prelude still holds the exact source endpoint.
    np.testing.assert_array_equal(
        tuned[:5, [1, 4, 16, 19]],
        np.repeat(original[0, [1, 4, 16, 19]][None, :], 5, axis=0),
    )
    assert np.all(tuned[5:, [1, 4, 16, 19]] <= cap)
    np.testing.assert_array_equal(
        tuned[5:, [2, 3, 17, 18]],
        original[1:, [2, 3, 17, 18]],
    )
    manifest = json.loads(
        (tmp_path / "manifest.json").read_text(encoding="utf-8")
    )
    override = manifest["primitives"][0]["outer_finger_hold_override"]
    assert override["indices"] == [1, 4, 16, 19]
    assert override["max_sdk_value"] == cap
