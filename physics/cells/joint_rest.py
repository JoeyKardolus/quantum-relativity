"""Joint rest volume: the size a joint swells back to when nothing pulls on it (bible rule 8).

Contract: `joint_rest(volume, joints, share)` returns one rest volume per joint (a, b):
share * (volume_a + volume_b) / 2, a share of its two energy volumes' mean volume.
It is read from the energy volumes each time, never stored: a joint between two
small, packed volumes rests small too.
"""
from __future__ import annotations

import numpy as np


def joint_rest(volume: np.ndarray, joints: np.ndarray, share: float) -> np.ndarray:
    """Rest volume per joint: `share` of the mean volume of the two energy volumes it separates."""
    return share * (volume[joints[:, 0]] + volume[joints[:, 1]]) / 2
