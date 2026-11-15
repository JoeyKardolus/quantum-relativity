"""Joint squeeze: how hard the pull presses each joint (bible rule 7).

Contract: `joint_squeeze(state, potential, attraction)` returns one value per volume:
for a joint, attraction times the mean pull potential of the energy volumes it
touches (0 for a joint touching none); 0 for every energy volume. The joint feels a
squeezing force squeeze * size against its push (swing_joints.py).
"""
from __future__ import annotations

import numpy as np

from physics.abab.state import State


def joint_squeeze(state: State, potential: np.ndarray, attraction: float) -> np.ndarray:
    """Per joint, attraction * mean potential of its energy volumes; 0 for energy volumes."""
    count = len(state.kind)
    volume, joint = state.contacts[:, 0], state.contacts[:, 1]
    touched = np.bincount(joint, minlength=count).astype(float)
    mean = np.divide(np.bincount(joint, potential[volume], count), touched, out=np.zeros(count),
                     where=touched > 0)
    return attraction * mean
