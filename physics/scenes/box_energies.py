"""Box energies: the pulling and pushing energy per Planck box of a universe, on the host, as floats.

Contract: `box_energies(universe, index=0)` returns (a_energy, b_energy), two float64
numpy arrays of shape (side, side, side) for universe `index` of a physics.engine
State: each box's whole quanta in the array of its sign (plus: pulling energy, an
energy volume's box, A; minus: pushing energy, a joint's box, B, as a positive count),
0 in the other and in empty boxes. Copies the quanta from the device once; the universe
is not changed.
"""
from __future__ import annotations

import numpy as np

from physics.engine import State, host_boxes


def box_energies(universe: State, index: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """(A-energy, B-energy) per box as float64 host arrays, split by the box's sign."""
    quanta = host_boxes(universe, index)["quanta"].astype(np.float64)
    return np.maximum(quanta, 0.0), np.maximum(-quanta, 0.0)
