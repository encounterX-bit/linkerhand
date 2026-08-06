"""HumanEgo-style adapters for LinkerHand experiments.

The package keeps learned-policy outputs separate from the existing camera
teleop stack.  Its public surface converts compact two-finger actions into the
repo's canonical 20-joint LinkerHand radian command.
"""

from .two_finger import (
    ACTION_DIMS,
    TWO_FINGER_IDX,
    TwoFingerConfig,
    action_to_l20,
    lock_candidate_to_two_finger,
    clip_l20_command,
    demo_actions,
    iter_action_records,
)

__all__ = [
    "ACTION_DIMS",
    "TWO_FINGER_IDX",
    "TwoFingerConfig",
    "action_to_l20",
    "lock_candidate_to_two_finger",
    "clip_l20_command",
    "demo_actions",
    "iter_action_records",
]
