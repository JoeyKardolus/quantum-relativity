"""Lumps start: an even universe with lumps placed in it, laid at rest in balance.

Contract: `lumps_start(settings, side, lumps, seed=1, device="cuda")` returns a fresh
universe (`new_universe` with `unevenness` 0, so every volume starts alike) in which
every box of every energy volume holds `1 + sum over lumps of peak * exp(-r ** 2 / (2
width ** 2))` times what it held (whole quanta), r the box's distance from the lump's
centre the short way round the wrapping lattice, a volume being 6 boxes wide. A lump
with a kick is set moving: every box's phase falls by `kick . (place - centre)` times a
window `exp(-r ** 2 / (2 (2.5 width) ** 2))`, so the fluid of the lump streams the way
the kick points and the fluid far from it is left as it was. The swirls stay: a box
keeps its share of its cell's swirl. Then `lay_start` lays the start again: sizes, the
pull at rest for the lumps, every joint in balance, the books.
It breaks "nothing is placed by hand" on purpose, as every placed scene does.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from physics.engine import ENERGY_VOLUME, Grid, Settings, lay_start, new_universe

_VOLUME_BOXES = 6


def _short_way(place: np.ndarray, centre: float, wide: int) -> np.ndarray:
    """A coordinate's distance from a centre the short way round a box `wide` across, signed."""
    return (place - centre + wide / 2.0) % wide - wide / 2.0


def lumps_start(settings: Settings, side: int, lumps: list, seed: int = 1, device: str = "cuda") -> Grid:
    """An even universe with the lumps placed, the kicked ones moving, at rest in balance."""
    settings = replace(settings, unevenness=0.0)
    universe = new_universe(settings, side, seed, device=device)
    volumes = np.flatnonzero(universe.host("kind") == ENERGY_VOLUME)
    wide = (side // 2) * _VOLUME_BOXES
    inside = np.arange(_VOLUME_BOXES)
    corner = np.stack([volumes // (side * side), (volumes // side) % side, volumes % side], axis=1) // 2 * _VOLUME_BOXES
    place = [corner[:, axis][:, None, None, None] + inside.reshape([-1 if one == axis else 1 for one in range(3)])[None]
             for axis in range(3)]
    quanta, phase = universe.quanta.numpy(), universe.phase.numpy()
    fill = np.ones((len(volumes), _VOLUME_BOXES, _VOLUME_BOXES, _VOLUME_BOXES))
    turn = np.zeros_like(fill)
    for lump in lumps:
        away = [_short_way(place[axis], lump.centre[axis], wide) for axis in range(3)]
        distance_squared = away[0] ** 2 + away[1] ** 2 + away[2] ** 2
        fill = fill + lump.peak * np.exp(-distance_squared / (2.0 * lump.width ** 2))
        along = sum(lump.kick[axis] * away[axis] for axis in range(3))
        turn = turn - along * np.exp(-distance_squared / (2.0 * (2.5 * lump.width) ** 2))
    quanta[volumes] = np.rint(quanta[volumes] * fill).astype(np.int32)
    phase[volumes] = np.angle(np.exp(1j * (phase[volumes] + turn)))
    universe.quanta.assign(quanta)
    universe.phase.assign(phase)
    lay_start(universe, settings)
    return universe
