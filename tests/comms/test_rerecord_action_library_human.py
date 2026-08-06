import json
from pathlib import Path

import numpy as np
import pytest

from src.comms.action_library import (
    ActionLibrary,
    FEATURE_PROFILE_NO_FINGER_SPLAY,
    G20_OPEN_POSE,
)
from src.comms.rerecord_action_library_human import (
    _complete_validation_takes,
    _targets,
    install_replacements,
    validate_replacements,
)


def _hand(thumb_shift: float = 0.0) -> np.ndarray:
    points = np.zeros((21, 3), dtype=np.float32)
    roots = ((1, -0.8), (5, -0.4), (9, 0.0), (13, 0.4), (17, 0.8))
    tips = (4, 8, 12, 16, 20)
    for (root, lateral), tip in zip(roots, tips):
        points[root:tip + 1, 1] = lateral
        points[root:tip + 1, 2] = np.linspace(0.3, 1.5, tip - root + 1)
    points[1:5, 0] += thumb_shift
    return points


def _takes(thumb_shift: float) -> list[np.ndarray]:
    hand = _hand(thumb_shift)
    return [np.stack([hand] * 8), np.stack([hand] * 10)]


def test_targets_can_select_one_action_without_reordering_manifest_records():
    manifest = {
        "primitives": [
            {"id": 2, "name": "two"},
            {"id": 3, "name": "three"},
            {"id": 1, "name": "one"},
        ]
    }

    selected = _targets(manifest, [3])

    assert [target.primitive_id for target in selected] == [3]
    assert selected[0].record is manifest["primitives"][1]


def test_targets_reject_unknown_action_id():
    manifest = {"primitives": [{"id": 3, "name": "three"}]}

    with pytest.raises(ValueError, match=r"unknown action IDs \[9\]"):
        _targets(manifest, [9])


def test_partial_rerecord_validation_keeps_unselected_library_takes(tmp_path: Path):
    old_take = np.stack([_hand(0.0)] * 6)
    np.save(tmp_path / "old_take.npy", old_take, allow_pickle=False)
    replacements = {3: _takes(0.4)}
    manifest = {
        "primitives": [
            {"id": 1, "human_templates": ["old_take.npy"]},
            {"id": 3, "human_templates": ["unused_after_replacement.npy"]},
        ]
    }

    combined = _complete_validation_takes(tmp_path, manifest, replacements)

    assert set(combined) == {1, 3}
    assert np.array_equal(combined[1][0], old_take)
    assert combined[3] is replacements[3]


def test_replacement_validation_separates_complete_new_take_set():
    result = validate_replacements(
        {1: _takes(0.0), 2: _takes(0.8)},
        feature_profile=FEATURE_PROFILE_NO_FINGER_SPLAY,
        margin=0.015,
    )

    assert result.correct == result.total == 4
    assert not result.misses


def test_install_replacements_preserves_robot_trajectories(tmp_path: Path):
    library = tmp_path / "library"
    library.mkdir()
    records = []
    robot_before = {}
    for primitive_id, shift in ((1, 0.0), (2, 0.8)):
        folder = library / f"primitive_{primitive_id:03d}_action"
        folder.mkdir()
        trajectory = np.stack((G20_OPEN_POSE, G20_OPEN_POSE - primitive_id))
        trajectory[:, 11:15] = 255
        robot_path = folder / "robot_trajectory.npy"
        np.save(robot_path, trajectory, allow_pickle=False)
        robot_before[primitive_id] = robot_path.read_bytes()
        old = np.stack([_hand(shift)] * 6)
        for take_index in range(2):
            np.save(folder / f"human_take_{take_index:03d}.npy", old, allow_pickle=False)
        records.append({
            "id": primitive_id,
            "name": f"action_{primitive_id}",
            "robot_trajectory": str(robot_path.relative_to(library)),
            "human_templates": [
                str((folder / f"human_take_{index:03d}.npy").relative_to(library))
                for index in range(2)
            ],
            "threshold": 0.1,
        })
    manifest = {
        "schema": ActionLibrary.SCHEMA,
        "feature_profile": FEATURE_PROFILE_NO_FINGER_SPLAY,
        "fps": 30.0,
        "primitives": records,
    }
    (library / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    replacements = {1: _takes(0.0), 2: _takes(0.8)}
    validation = validate_replacements(
        replacements,
        feature_profile=FEATURE_PROFILE_NO_FINGER_SPLAY,
        margin=0.015,
    )
    session = tmp_path / "session"
    session.mkdir()

    archive = install_replacements(
        library,
        manifest,
        _targets(manifest),
        replacements,
        validation,
        session_dir=session,
    )

    loaded = ActionLibrary.load(library)
    assert archive.joinpath("manifest.json").is_file()
    for primitive_id, primitive in loaded.primitives.items():
        robot_path = library / records[primitive_id - 1]["robot_trajectory"]
        assert robot_path.read_bytes() == robot_before[primitive_id]
        assert len(primitive.templates) == 2
        assert primitive.templates[0].shape[0] == 8
        assert archive.joinpath(
            f"primitive_{primitive_id:03d}_action", "human_take_000.npy"
        ).is_file()
