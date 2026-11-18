"""Pair run: two placed lumps one distance apart, and the first lump alone, stepped side by side.

Contract: `pair_run(cells_per_side, gap, factor, steps, settle, seed, settings, device)`
builds two placed starts (packed_start): an even grid with two volumes packed `factor`
times, `gap` volumes apart along the first axis, and the same grid with only the first.
Both take `steps` steps with every rule on. Returns a dict of floats:
  distance      summed lengths of the cells between the two lumps' middles at the start
  reach         half the summed lengths between the middles of the two volumes beside
                the first lump, at the start; round: once round the lumps' row
  tilt          how much deeper the pull lies on the second lump's side of the first:
                (pull at the volume beside it away from the second - pull at the volume
                beside it toward the second) over the summed lengths between those two
                volumes' middles, the mean over the steps after `settle`
  tilt_alone    the same at the same place with the second lump not there
  closing       how far the two lumps' extra energy moved toward each other over the
                run, in volumes: per lump the mean place (extra_centre) of the energy
                above an even volume's within gap // 2 volumes of it, in a tube 5
                volumes wide; end minus start, the pair's sum
  closing_alone the first lump's own shift with the second not there
  kept          the share of the lumps' extra energy still in the two lump volumes at the end
"""
from __future__ import annotations

import numpy as np

from physics.engine import Grid, Settings, step
from physics.scenes.along_grid.definitions.row_cells import row_cells
from physics.scenes.along_grid.logic.middle_distances import middle_distances
from physics.scenes.placed_start.logic.packed_start import packed_start
from physics.scenes.run.definitions.even_volume_quanta import even_volume_quanta
from physics.scenes.two_lumps.definitions.extra_centre import extra_centre
from physics.scenes.volumes.logic.volume_field import volume_field

_TUBE = 2


def _row_cell(universe: Grid, first: int, second: int, third: int) -> int:
    """The id of a lattice place (wrapping)."""
    side = universe.side
    return ((first % side) * side + second % side) * side + third % side


def _between(universe: Grid, low: int, high: int, middle: int) -> float:
    """Summed lengths from the middle of place `low` to the middle of place `high` along the first axis."""
    row = row_cells(universe.side, _row_cell(universe, low, middle, middle), 0, 1, high - low)
    return float(middle_distances(universe, row)[-1])


def _tilt(universe: Grid, place: tuple) -> float:
    """How much deeper the pull lies on the far side of a lump along the first axis, per length between its two neighbour volumes."""
    pull = universe.host("pull")
    near = _row_cell(universe, place[0] + 2, place[1], place[2])
    far = _row_cell(universe, place[0] - 2, place[1], place[2])
    return float(pull[far] - pull[near]) / _between(universe, place[0] - 2, place[0] + 2, place[1])


def _centre(universe: Grid, place: tuple, gap: int, even: float) -> float:
    """Mean place along the axis, in volumes from the lump, of the energy above an even volume's near the lump."""
    held = volume_field(universe)[0]
    across = held.shape[0]
    offsets = np.arange(-(gap // 2), gap // 2 + 1)
    tube = np.arange(-_TUBE, _TUBE + 1)
    block = held[np.ix_((place[0] // 2 + offsets) % across, (place[1] // 2 + tube) % across,
                        (place[2] // 2 + tube) % across)]
    return extra_centre(np.maximum(block - even, 0.0).sum(axis=(1, 2)), offsets)


def pair_run(cells_per_side: int, gap: int, factor: float, steps: int, settle: int, seed: int, settings: Settings,
             device: str) -> dict:
    """One distance: the pair and the first lump alone, stepped side by side in time."""
    middle = (cells_per_side // 4) * 2
    low = middle - 2 * (gap // 2)
    places = [(low, middle, middle), (low + 2 * gap, middle, middle)]
    pair = packed_start(cells_per_side, {place: factor for place in places}, seed, settings, device)
    alone = packed_start(cells_per_side, {places[0]: factor}, seed, settings, device)
    even = even_volume_quanta(settings)
    read = ((pair, places[0]), (pair, places[1]), (alone, places[0]))
    fixed = {"distance": _between(pair, places[0][0], places[1][0], middle),
             "reach": 0.5 * _between(pair, places[0][0] - 2, places[0][0] + 2, middle),
             "round": _between(pair, 0, cells_per_side, middle)}
    before = [_centre(universe, place, gap, even) for universe, place in read]
    tilts, lone = [], []
    for taken in range(1, steps + 1):
        step(pair, settings)
        step(alone, settings)
        if taken > settle:
            tilts.append(_tilt(pair, places[0]))
            lone.append(_tilt(alone, places[0]))
    after = [_centre(universe, place, gap, even) for universe, place in read]
    held = volume_field(pair)[0]
    extra = sum(float(held[place[0] // 2, place[1] // 2, place[2] // 2]) - even for place in places)
    return {**fixed, "tilt": float(np.mean(tilts)) if tilts else 0.0, "tilt_alone": float(np.mean(lone)) if lone else 0.0,
            "closing": float((after[0] - before[0]) - (after[1] - before[1])),
            "closing_alone": float(after[2] - before[2]), "kept": extra / (2.0 * (factor - 1.0) * even)}
