"""The energy-length rule (bible rule 4): length = LENGTH_SCALE / energy.

Contract: `energy_length(value)` turns an energy into a length, or a length into
an energy; the rule is its own inverse, so one function does both ways. Works
elementwise on numpy arrays. LENGTH_SCALE is the one constant of the rule (the
analogue of h*c in E = h*c / wavelength); every builder and rule uses it.
"""
from __future__ import annotations

import numpy as np

#: energy * length of every connection when the energy-length rule holds exactly.
LENGTH_SCALE = 1.0


def energy_length(value: np.ndarray | float) -> np.ndarray | float:
    """Length of a connection with this energy, or energy of one with this length."""
    return LENGTH_SCALE / np.asarray(value, dtype=float)
