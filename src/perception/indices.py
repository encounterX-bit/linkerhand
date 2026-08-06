"""MediaPipe Hands 21-landmark index convention (shared by the whole module).

This is the canonical landmark numbering used by the ``hand_landmarks`` contract
and by ADR-0003's segment convention. Do NOT re-number these anywhere else.
"""
from __future__ import annotations

import numpy as np

N_LANDMARKS = 21

WRIST = 0
# thumb: CMC, MCP, IP, TIP
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20

# finger -> (a, b, c, d) landmark indices, exactly ADR-0003 Table.
FINGER_LANDMARKS = {
    "thumb": (THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP),
    "index": (INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP),
    "middle": (MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP),
    "ring": (RING_MCP, RING_PIP, RING_DIP, RING_TIP),
    "little": (PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP),
}

# The four non-thumb MCPs plus the wrist span the palm. In the L20/oracle
# hand_base frame these five points are (pose-invariantly) coplanar, so they
# define the frame robustly. See frame.py / ADR-0003.
PALM_LANDMARKS = (WRIST, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP)

# A canonical, idealised RIGHT flat hand expressed directly in the hand_base
# frame (+z = fingers, +y = radial/thumb side, +x = dorsal palm normal). Used to
# seed synthetic streams and the committed synthetic "real" fixture. These are
# generic hand-geometry constants, not robot/oracle internals.
CANONICAL_FLAT_RIGHT = np.array(
    [
        [0.000, 0.000, 0.000],   # 0  wrist
        [0.000, 0.035, 0.030],   # 1  thumb CMC
        [0.027, 0.053, 0.062],   # 2  thumb MCP
        [0.046, 0.065, 0.084],   # 3  thumb IP
        [0.061, 0.075, 0.102],   # 4  thumb TIP
        [0.000, 0.022, 0.090],   # 5  index MCP
        [0.000, 0.022, 0.130],   # 6  index PIP
        [0.000, 0.022, 0.155],   # 7  index DIP
        [0.000, 0.022, 0.175],   # 8  index TIP
        [0.000, 0.006, 0.090],   # 9  middle MCP
        [0.000, 0.006, 0.135],   # 10 middle PIP
        [0.000, 0.006, 0.163],   # 11 middle DIP
        [0.000, 0.006, 0.185],   # 12 middle TIP
        [0.000, -0.012, 0.090],  # 13 ring MCP
        [0.000, -0.012, 0.130],  # 14 ring PIP
        [0.000, -0.012, 0.156],  # 15 ring DIP
        [0.000, -0.012, 0.177],  # 16 ring TIP
        [0.000, -0.030, 0.090],  # 17 pinky MCP
        [0.000, -0.030, 0.122],  # 18 pinky PIP
        [0.000, -0.030, 0.142],  # 19 pinky DIP
        [0.000, -0.030, 0.160],  # 20 pinky TIP
    ],
    dtype=float,
)
