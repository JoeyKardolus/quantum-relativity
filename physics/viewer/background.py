"""The far background of a scene: the joints its size is read from.

Contract: `background_joints(network, marked, far)` returns the joints whose
network distance from every marked joint is at least `far` mean connection
lengths (all joints when nothing is marked). Its mean connection length is
the ruler that tells a lump closing in from all of space shrinking. Raises
ValueError when no joint is that far, since there is then no background.
"""
from __future__ import annotations

import numpy as np

from physics.measure import distances, mean_length
from physics.network import Network


def background_joints(network: Network, marked: tuple[int, ...], far: float) -> np.ndarray:
    """Joints at least `far` mean lengths from every marked joint."""
    if not marked:
        return np.arange(network.joint_count)
    reach = distances(network, list(marked)).min(axis=0) / mean_length(network)
    joints = np.flatnonzero(reach >= far)
    if not len(joints):
        raise ValueError(f"no joint lies {far} mean lengths from every marked joint")
    return joints
