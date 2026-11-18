"""Power spectrum: how an energy contrast is shared over sizes (clustering spectrum).

Contract: `power_spectrum(density)` takes a float array of energy per place, shape (side,
side, side), and returns (wavenumbers, power): two float64 arrays over the shells
k = 1 .. side // 2 (cycles per cube side; wavelength side / k places). The field is the
contrast density / mean - 1 over the whole cube, Fourier transformed as periodic (numpy
rfftn; the universe wraps); power is the mean |mode|**2 / side**3 over the modes whose
|k| rounds to the shell (each rfft half-mode counted twice, its mirror). An even field
gives 0 everywhere. Raises ValueError for a non-cube, side < 4 or a field holding no
energy.
"""
from __future__ import annotations

import numpy as np


def power_spectrum(density: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Shell-averaged power of the density contrast for k = 1 .. side // 2."""
    side = density.shape[0]
    if density.shape != (side, side, side) or side < 4:
        raise ValueError(f"density must be a cube of side >= 4, got {density.shape}")
    mean = float(density.mean())
    if mean <= 0.0:
        raise ValueError("density holds no energy")
    modes = np.fft.rfftn(density / mean - 1.0)
    power = np.abs(modes) ** 2 / side ** 3
    full = np.fft.fftfreq(side) * side
    half = np.fft.rfftfreq(side) * side
    shell = np.rint(np.sqrt(full[:, None, None] ** 2 + full[None, :, None] ** 2
                            + half[None, None, :] ** 2)).astype(int)
    weight = np.where(half[None, None, :] > 0, 2.0, 1.0) * np.ones_like(power)
    wavenumbers = np.arange(1, side // 2 + 1)
    counts = np.bincount(shell.ravel(), weights=weight.ravel(), minlength=side)
    sums = np.bincount(shell.ravel(), weights=(power * weight).ravel(), minlength=side)
    return wavenumbers.astype(np.float64), sums[wavenumbers] / counts[wavenumbers]
