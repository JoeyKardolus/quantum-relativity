"""Uneven field: a seeded Gaussian random field with ripples at every size, by FFT.

Contract: `uneven_field(side, seed, slope, unevenness)` returns a float64 numpy array
of shape (side, side, side), periodic over the cube, with mean 0 and root mean square
exactly `unevenness`. Its power spectrum follows P(k) = k**slope: white noise
(numpy default_rng(seed), standard normal per box) is Fourier transformed (numpy
rfftn, once on the CPU; Warp has no FFT), every mode of wavenumber k (cycles per
cube side, 1 to the Nyquist corner) is scaled by k**(slope / 2), the k = 0 mode is
removed, and the result is transformed back and scaled to the asked root mean square.
So `slope` says how the amplitude is shared over wavelengths: in 3D, slope -3 gives
every octave of wavelength the same share of the unevenness (ripples at all sizes
alike), a slope above -3 gives the small ripples more, below -3 the large zones more.
Raises ValueError for side < 8, a non-finite slope, or unevenness < 0 or not finite.
"""
from __future__ import annotations

import math

import numpy as np


def uneven_field(side: int, seed: int, slope: float, unevenness: float) -> np.ndarray:
    """A periodic Gaussian random field with power spectrum k**slope and rms unevenness."""
    if side < 8:
        raise ValueError(f"side must be at least 8, got {side}")
    if not math.isfinite(slope):
        raise ValueError(f"slope must be finite, got {slope}")
    if not (math.isfinite(unevenness) and unevenness >= 0):
        raise ValueError(f"unevenness must be finite and >= 0, got {unevenness}")
    noise = np.random.default_rng(seed).standard_normal((side, side, side))
    modes = np.fft.rfftn(noise)
    full = np.fft.fftfreq(side) * side
    half = np.fft.rfftfreq(side) * side
    wavenumber = np.sqrt(full[:, None, None] ** 2 + full[None, :, None] ** 2 + half[None, None, :] ** 2)
    wavenumber[0, 0, 0] = 1.0
    modes *= wavenumber ** (slope / 2.0)
    modes[0, 0, 0] = 0.0
    field = np.fft.irfftn(modes, s=noise.shape, axes=(0, 1, 2))
    spread = float(np.sqrt(np.mean(field ** 2)))
    if spread == 0.0:
        return field
    return field * (unevenness / spread)
