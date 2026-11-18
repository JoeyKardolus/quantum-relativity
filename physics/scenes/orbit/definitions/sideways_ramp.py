"""Sideways ramp: the start that sets a volume's fluid moving across the line to the big one.

Contract: `SIDEWAYS` is pi / 6: the phase step per box, in radians, a twelfth of a turn
(half a turn across a block of 6). Chosen before any run; a start number for the fit to
search, not tuned to a result. `sideways_ramp(phase, step, axis)` takes one cell's
block of phases (float array, 3D) and returns a float64 copy with `step` times the box
index along `axis` added, wrapped into (-pi, pi]. In the engine fluid flows from the
wave whose phase is ahead, so a ramp is a flow along that axis; the cell's own swirl
stays wound underneath. Raises ValueError for another axis.
"""
from __future__ import annotations

import math

import numpy as np

SIDEWAYS = math.pi / 6.0


def sideways_ramp(phase: np.ndarray, step: float, axis: int) -> np.ndarray:
    """The block's phases with a ramp of `step` radians a box added along `axis`, wrapped."""
    if axis not in (0, 1, 2):
        raise ValueError(f"axis must be 0, 1 or 2, got {axis}")
    shape = [1, 1, 1]
    shape[axis] = phase.shape[axis]
    ramped = np.asarray(phase, dtype=np.float64) + step * np.arange(phase.shape[axis]).reshape(shape)
    return math.pi - np.mod(math.pi - ramped, 2.0 * math.pi)
