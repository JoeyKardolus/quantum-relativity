"""Joint energy: the energy a joint must hold to want a given size (joint_rest_size.py inverted).

Contract: `joint_energy(size)` returns, elementwise, JOINT_ENERGY * (JOINT_SIZE / size) ** (1/3),
so joint_rest_size(joint_energy(size)) is size (to rounding). split_volumes() uses it to price a
new joint at rest.
"""
from __future__ import annotations

import numpy as np

from physics.abab.joint_rest_size import JOINT_ENERGY, JOINT_SIZE


def joint_energy(size: np.ndarray | float) -> np.ndarray:
    """Per joint, the energy whose rest size is `size`."""
    return JOINT_ENERGY * np.cbrt(JOINT_SIZE / np.asarray(size, dtype=float))
