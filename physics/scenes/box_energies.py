"""Box energies: the A- and B-energy per Planck box of a universe, on the host, as floats.

Contract: `box_energies(universe)` returns (a_energy, b_energy), two float64 numpy arrays
of shape (side, side, side): each box's energy |psi|**2 (wave_field.py) in the array of
its owner's kind (energy volume: A, joint: B), 0 in the other and in empty boxes.
Copies the owners, the field and the kinds from the device once; the universe is not
changed.
"""
from __future__ import annotations

import numpy as np

from physics.potts import JOINT, Universe
from physics.scenes.wave_field import wave_field


def box_energies(universe: Universe) -> tuple[np.ndarray, np.ndarray]:
    """(A-energy, B-energy) per box as float64 host arrays, split by the owner's kind."""
    owner = universe.owner.numpy()
    energy = np.abs(wave_field(universe)) ** 2
    kind = universe.kind.numpy()
    joint = (owner >= 0) & (kind[np.maximum(owner, 0)] == JOINT)
    return np.where(joint, 0.0, energy), np.where(joint, energy, 0.0)
