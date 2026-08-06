from unittest.mock import patch

import numpy as np

from src.comms.visual_act_to_linkerhand import draw_status


def _drawn_text(*, message: str, minimal_overlay: bool) -> list[str]:
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    contact = {
        "finger_contact_count": 0,
        "minimum_fingers": 3,
        "thumb_contact": False,
        "success_gate_met": False,
        "touch_fresh": True,
        "continuous_contact_seconds": 0.0,
    }
    with patch(
        "src.comms.visual_act_to_linkerhand.cv2.putText",
        side_effect=lambda image, *_args, **_kwargs: image,
    ) as put_text:
        draw_status(
            frame,
            armed=True,
            motion_enabled=True,
            raw=np.asarray([42.0, 257.9], dtype=np.float32),
            message=message,
            contact_summary=contact,
            cube_ready_text="END TRACK depart=Y error=249.0/12 settle=0/10",
            minimal_overlay=minimal_overlay,
        )
    return [str(call.args[1]) for call in put_text.call_args_list]


def test_minimal_overlay_hides_routine_diagnostics() -> None:
    text = _drawn_text(message="max target delta 60.4; step <= 20", minimal_overlay=True)

    assert "ARMED / PUBLISHING" in text
    assert "SPACE arm/disarm | R reset | Q/ESC quit" in text
    assert not any(value.startswith("ACT raw min/max") for value in text)
    assert not any(value.startswith("TOUCH") for value in text)
    assert not any(value.startswith("END TRACK") for value in text)
    assert not any(value.startswith("max target delta") for value in text)


def test_minimal_overlay_keeps_safety_message() -> None:
    text = _drawn_text(message="ROS state stale; DISARMED", minimal_overlay=True)

    assert "ROS state stale; DISARMED" in text
