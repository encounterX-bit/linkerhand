import json

import numpy as np
import pytest

from src.finger_retarget.constants import ACTIVE_IDX, RESERVED_IDX
from src.humanego_linkerhand import (
    TWO_FINGER_IDX,
    TwoFingerConfig,
    action_to_l20,
    clip_l20_command,
    iter_action_records,
    lock_candidate_to_two_finger,
)


def test_pinch3_open_is_zero_command():
    out = action_to_l20([0.0, 0.0, 0.0], TwoFingerConfig(side="right"))

    assert out["side"] == "right"
    assert out["clamped"] is True
    assert out["active_idx"] == list(ACTIVE_IDX)
    assert out["joint_rad"] == [0.0] * 20


def test_pinch3_only_thumb_and_index_move():
    q = action_to_l20([1.0, 1.0, -0.5], TwoFingerConfig(side="right"))["joint_rad"]

    moving = {i for i, v in enumerate(q) if abs(v) > 1e-12}
    assert moving <= set(TWO_FINGER_IDX)
    assert q[0] > 0.0
    assert q[1] > 0.0
    assert q[6] < 0.0
    for idx in RESERVED_IDX:
        assert q[idx] == 0.0


def test_joint7_normalized_maps_all_two_finger_channels():
    cfg = TwoFingerConfig(side="right", mode="joint7")
    q = action_to_l20([1, 1, 1, 1, 1, 1, 1], cfg)["joint_rad"]

    assert q[0] == pytest.approx(cfg.thumb_base_max)
    assert q[5] == pytest.approx(cfg.thumb_abd_max)
    assert q[10] == pytest.approx(cfg.thumb_opp_max)
    assert q[15] == pytest.approx(cfg.thumb_tip_max)
    assert q[1] == pytest.approx(cfg.index_base_max)
    assert q[6] == pytest.approx(cfg.index_spread_max)
    assert q[16] == pytest.approx(cfg.index_tip_max)


def test_joint7_radians_are_clipped_to_safe_subset():
    cfg = TwoFingerConfig(side="right", mode="joint7", input_range="radians")
    q = action_to_l20([99, 99, 99, 99, 99, -99, 99], cfg)["joint_rad"]

    assert q[0] <= 0.79
    assert q[5] <= 1.22
    assert q[10] <= 1.4
    assert q[15] <= 1.05
    assert q[1] <= 1.4
    assert q[6] >= -0.17
    assert q[16] <= 1.57
    assert all(q[i] == 0.0 for i in set(ACTIVE_IDX) - set(TWO_FINGER_IDX))


def test_clip_rejects_nonfinite():
    q = np.zeros(20)
    q[0] = np.nan
    with pytest.raises(ValueError):
        clip_l20_command(q, "right")


def test_iter_action_records_jsonl(tmp_path):
    path = tmp_path / "actions.jsonl"
    path.write_text(
        json.dumps({"t": 0.1, "action": [0.2, 0.3, 0.4]}) + "\n"
        + json.dumps([0.5, 0.6, 0.7]) + "\n",
        encoding="utf-8",
    )

    records = list(iter_action_records(path))
    assert records == [([0.2, 0.3, 0.4], 0.1), ([0.5, 0.6, 0.7], None)]


def test_lock_candidate_to_two_finger_preserves_contract_shape():
    candidate = {
        "side": "right",
        "joint_rad": [0.5] * 20,
        "active_idx": list(ACTIVE_IDX),
        "clamped": True,
        "t": 1.25,
    }

    out = lock_candidate_to_two_finger(candidate)

    assert out["side"] == "right"
    assert out["t"] == 1.25
    assert out["clamped"] is True
    assert out["active_idx"] == list(ACTIVE_IDX)
    assert set(i for i, v in enumerate(out["joint_rad"]) if abs(v) > 1e-12) <= set(TWO_FINGER_IDX)
    assert all(out["joint_rad"][i] == 0.0 for i in RESERVED_IDX)
