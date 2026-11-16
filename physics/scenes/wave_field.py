"""Wave field: the energy fluid's one complex number per Planck box, on the host.

Contract: `wave_field(universe, index=0)` copies universe `index` of a physics.engine
State from the device once and returns its fluid as a complex128 numpy array of shape
(side, side, side): psi = sqrt(|quanta|) exp(i phase) (the engine never stores it, it
is tied to the whole quanta, physics/engine/state.py), so |psi|**2 is the box's whole
quanta ignoring sign, its flow comes from how the phase turns from box to box, and an
empty box holds 0. The universe is not changed.
"""
from __future__ import annotations

import numpy as np

from physics.engine import State, host_boxes


def wave_field(universe: State, index: int = 0) -> np.ndarray:
    """psi per box as a complex host array."""
    boxes = host_boxes(universe, index)
    size = np.sqrt(np.abs(boxes["quanta"].astype(np.float64)))
    return size * np.exp(1j * boxes["phase"].astype(np.float64))
