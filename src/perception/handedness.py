"""Map an estimator's camera-view handedness label to the L20 side convention.

MediaPipe assigns "Left"/"Right" *assuming the input image is mirrored* (a
selfie / front-facing camera with the frame flipped horizontally). If the image
is NOT mirrored, the reported label refers to the opposite physical hand and must
be swapped. The L20 ``/cb_*_hand_*`` topics and the ``hand_landmarks`` contract
use the physical side ("left"/"right"), so we resolve to that here.

This is the perception-side handedness trap: get the swap wrong and a left hand
drives the right-hand solver (mirrored chirality), silently.
"""
from __future__ import annotations

_VALID = ("left", "right")


def to_l20_side(camera_label: str, image_mirrored: bool = False) -> str:
    """Return the physical L20 side ("left"/"right").

    Parameters
    ----------
    camera_label : the estimator's raw label, e.g. MediaPipe "Left"/"Right".
    image_mirrored : True if the frame fed to the estimator was horizontally
        mirrored (selfie view). When False, the label is swapped.
    """
    if camera_label is None:
        raise ValueError("camera_label is None (no handedness from estimator)")
    label = camera_label.strip().lower()
    if label not in _VALID:
        raise ValueError(f"unknown handedness label {camera_label!r}")
    if image_mirrored:
        return label
    return "right" if label == "left" else "left"
