import json
from pathlib import Path

import numpy as np
import pytest

from src.comms.action_library import ActionLibrary
from src.comms.clone_reversed_action import clone_reversed_action


def _make_library(root: Path) -> tuple[np.ndarray, list[np.ndarray]]:
    source = root / "primitive_001_close"
    source.mkdir(parents=True)
    trajectory = np.stack(
        (
            np.full(20, 255, dtype=np.float32),
            np.full(20, 180, dtype=np.float32),
            np.full(20, 100, dtype=np.float32),
        )
    )
    trajectory[:, 11:15] = 255
    templates = [
        np.arange(length * 21 * 3, dtype=np.float32).reshape(length, 21, 3)
        for length in (6, 8)
    ]
    np.save(source / "robot_trajectory.npy", trajectory, allow_pickle=False)
    template_paths = []
    for index, template in enumerate(templates):
        path = source / f"human_take_{index:03d}.npy"
        np.save(path, template, allow_pickle=False)
        template_paths.append(str(path.relative_to(root)))
    manifest = {
        "schema": ActionLibrary.SCHEMA,
        "fps": 30.0,
        "primitives": [
            {
                "id": 1,
                "name": "close",
                "robot_trajectory": str(
                    (source / "robot_trajectory.npy").relative_to(root)
                ),
                "human_templates": template_paths,
                "threshold": 0.08,
                "cooldown_frames": 20,
            }
        ],
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    return trajectory, templates


def test_clone_reversed_action_reverses_robot_and_human_time(tmp_path):
    trajectory, templates = _make_library(tmp_path)

    result = clone_reversed_action(
        tmp_path,
        source_id=1,
        primitive_id=6,
        name="open_reverse_one",
    )

    library = ActionLibrary.load(tmp_path)
    reverse = library.primitives[6]
    source = library.primitives[1]
    np.testing.assert_array_equal(reverse.trajectory, trajectory[::-1])
    assert reverse.manual_from_start is True
    for actual, expected in zip(reverse.templates, source.templates):
        np.testing.assert_array_equal(actual, expected[::-1])
    assert result["trajectory_frames"] == 3
    assert result["templates"] == 2
    np.testing.assert_array_equal(
        np.load(
            tmp_path / "primitive_001_close" / "robot_trajectory.npy",
            allow_pickle=False,
        ),
        trajectory,
    )


def test_clone_reversed_action_refuses_existing_destination_id(tmp_path):
    _make_library(tmp_path)
    clone_reversed_action(
        tmp_path,
        source_id=1,
        primitive_id=6,
        name="open_reverse_one",
    )

    with pytest.raises(ValueError, match="already exists"):
        clone_reversed_action(
            tmp_path,
            source_id=1,
            primitive_id=6,
            name="another",
        )
