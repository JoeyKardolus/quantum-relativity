"""Box field: the depth of a 1 / distance pull from one point at every volume of a wrapping cube.

Contract: `box_field(across, fine=4)` returns a float64 array (across, across, across):
entry [i, j, k] is the depth of the pull of a point at the origin, at the volume i, j, k
volumes away (wrapping), above its depth at the farthest place (across // 2 on every
axis), which reads 0, in the unit in which a lone point's pull is 1 / distance in
volumes (the unit box_depth uses). It is the wrapping cube's own answer to "a pull with no preferred
direction": Poisson's equation with a point source solved by Fourier transform on a
grid `fine` times finer, the even level removed. The cube has axes, so the field is not
the same at two places equally far from the origin on and off an axis; a scene compares
the engine's pull with this, not with 1. Raises ValueError for across < 4 or fine < 1.
"""
from __future__ import annotations

import numpy as np


def box_field(across: int, fine: int = 4) -> np.ndarray:
    """Depth of a point's 1 / distance pull per volume of a wrapping cube, 0 at the farthest place."""
    if across < 4 or fine < 1:
        raise ValueError(f"need across >= 4 and fine >= 1, got {across} and {fine}")
    side = across * fine
    wave = 2.0 * np.pi * np.fft.fftfreq(side, d=1.0 / fine)
    square = wave[:, None, None] ** 2 + wave[None, :, None] ** 2 + wave[None, None, :] ** 2
    square[0, 0, 0] = 1.0
    source = np.zeros((side,) * 3)
    source[0, 0, 0] = 1.0
    spectrum = np.fft.fftn(source) / square
    spectrum[0, 0, 0] = 0.0
    depth = 4.0 * np.pi * fine ** 3 * np.fft.ifftn(spectrum).real[::fine, ::fine, ::fine]
    return depth - depth[across // 2, across // 2, across // 2]
