"""Uneven field: how the start's energy differs from energy volume to energy volume.

Contract: `uneven_field(side, seed, slope, unevenness)` (host numpy) returns a float64
array of shape (side / 2, side / 2, side / 2): one value per energy volume of a start
grid with `side` cells along an edge (the volumes sit where all three indices are even,
so value [i, j, k] is the volume at (2 i, 2 j, 2 k), which is the volumes in rising id
order). The field wraps round like the grid, has mean 0 and root mean square
`unevenness`, and is the same for the same seed.
With 8 or more volumes along an edge it has ripples at every size: seeded white noise
(one normal per volume from `draw_uniform`, passes `START_FIELD_A` and `START_FIELD_B`,
the volume's place in the field as index) is Fourier transformed, every ripple of wavenumber k (cycles
per edge) is scaled by k ** (slope / 2), the even level is removed and the result is
scaled to the asked root mean square. `slope` -3 gives every octave of ripple size the
same share; above -3 the small ripples get more, below it the large zones.
With fewer than 8 volumes along an edge there is no room for ripples: it is plain seeded
noise with its even level removed, scaled to the same root mean square.
Raises ValueError for a side that is odd or below 4, a slope that is not a number, or an
unevenness below 0.
"""
from __future__ import annotations

import math

import numpy as np

from physics.engine.axiom_1_energy.definitions.draw_uniform import draw_uniform_rows
from physics.engine.axiom_1_energy.definitions.passes import START_FIELD_A, START_FIELD_B

_RIPPLE_SIDE = 8


def _normals(side: int, seed: int) -> np.ndarray:
    """One standard normal per volume, from the engine's one source of draws (Box and Muller's two draws each)."""
    places = np.arange(side ** 3)
    first = draw_uniform_rows(seed, 0, START_FIELD_A, places)
    second = draw_uniform_rows(seed, 0, START_FIELD_B, places)
    return (np.sqrt(-2.0 * np.log(1.0 - first)) * np.cos(2.0 * np.pi * second)).reshape(side, side, side)


def _rippled(side: int, seed: int, slope: float, unevenness: float) -> np.ndarray:
    """A wrapping random field of `side` values an edge with power k ** slope, scaled to the asked spread."""
    modes = np.fft.rfftn(_normals(side, seed))
    full = np.fft.fftfreq(side) * side
    half = np.fft.rfftfreq(side) * side
    wavenumber = np.sqrt(full[:, None, None] ** 2 + full[None, :, None] ** 2 + half[None, None, :] ** 2)
    wavenumber[0, 0, 0] = 1.0
    modes *= np.power(wavenumber, slope / 2.0)
    modes[0, 0, 0] = 0.0
    field = np.fft.irfftn(modes, s=(side, side, side), axes=(0, 1, 2))
    spread = float(np.sqrt(np.mean(field ** 2)))
    if spread == 0.0:
        return field
    return field * (unevenness / spread)


def _plain(side: int, seed: int, unevenness: float) -> np.ndarray:
    """Seeded noise with no ripples, mean 0 and the asked spread, for a lattice too small to hold any."""
    field = _normals(side, seed)
    field = field - field.mean()
    spread = float(np.sqrt(np.mean(field ** 2)))
    if spread == 0.0:
        return field
    return field * (unevenness / spread)


def uneven_field(side: int, seed: int, slope: float, unevenness: float) -> np.ndarray:
    """The seeded field on the volume lattice of a grid with `side` cells an edge."""
    if side < 4 or side % 2:
        raise ValueError(f"side must be even and at least 4, got {side}")
    if not math.isfinite(slope):
        raise ValueError(f"slope must be a number, got {slope}")
    if not (math.isfinite(unevenness) and unevenness >= 0.0):
        raise ValueError(f"unevenness must be 0 or more, got {unevenness}")
    volumes = side // 2
    if volumes >= _RIPPLE_SIDE:
        return _rippled(volumes, seed, slope, unevenness)
    return _plain(volumes, seed, unevenness)
