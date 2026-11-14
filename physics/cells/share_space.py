"""Share space: hand space an energy volume gives up (or takes) to the joints around it.

Contract: `share_space(joints, joint_volume, space)` returns new joint volumes.
space (n,) is per energy volume the space it hands its joints (negative: takes
from them). Each energy volume splits its space over its own joints in
proportion to their volumes, so a thin joint gets or gives little. Every unit
handed lands in exactly one joint: sum(new) = sum(joint_volume) + sum(space over
energy volumes that have a joint). An energy volume with no joint keeps nothing:
its space is not handed anywhere (the caller counts it). Raises ValueError when a
joint would be left with no volume (an energy volume asked its joints for more
space than they hold).
"""
from __future__ import annotations

import numpy as np


def share_space(joints: np.ndarray, joint_volume: np.ndarray, space: np.ndarray) -> np.ndarray:
    """Joint volumes after every energy volume hands `space` to its joints by their size."""
    heads, tails = joints[:, 0], joints[:, 1]
    count = len(space)
    around = np.bincount(heads, joint_volume, count) + np.bincount(tails, joint_volume, count)
    rate = np.divide(space, around, out=np.zeros(count), where=around > 0)
    shared = joint_volume * (1 + rate[heads] + rate[tails])
    if (shared <= 0).any():
        raise ValueError("an energy volume took more space than its joints held")
    return shared
