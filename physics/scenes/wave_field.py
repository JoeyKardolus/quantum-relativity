"""Wave field: the energy fluid's one complex number per Planck box, on the host.

Contract: `wave_field(universe)` copies the Universe's `psi` (universe.py: vec2d per
box, real and imaginary part of the Gross-Pitaevskii fluid's number) from the device
once and returns it as a complex128 numpy array of shape (side, side, side): the box's
energy is its |psi|**2, its flow comes from how the phase turns from box to box, and
an empty box holds 0. The universe is not changed.
"""
from __future__ import annotations

import numpy as np

from physics.potts import Universe


def wave_field(universe: Universe) -> np.ndarray:
    """psi per box as a complex host array."""
    pairs = universe.psi.numpy().astype(np.float64)
    return pairs[..., 0] + 1j * pairs[..., 1]
