"""Joint rest thickness: the thickness a joint swings around when nothing pulls on it (bible rule 8).

Contract: `joint_rest(volume, joints, share)` returns one rest thickness per joint (a, b):
share * (volume_a ** (1/3) + volume_b ** (1/3)) / 2, a share of its two energy
volumes' mean length. It is read from the energy volumes each time, never stored:
a joint between two small, packed volumes rests thin too.
"""
from __future__ import annotations

import numpy as np


def joint_rest(volume: np.ndarray, joints: np.ndarray, share: float) -> np.ndarray:
    """Rest thickness per joint: `share` of the mean length of the two energy volumes it separates."""
    length = np.cbrt(volume)
    return share * (length[joints[:, 0]] + length[joints[:, 1]]) / 2
