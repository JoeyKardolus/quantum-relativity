"""Packed start: an even grid in which chosen energy volumes hold more energy (a placed start).

Contract: `packed_start(cells_per_side, packed, seed=1, settings=None, device="cuda",
start=None)` builds physics.engine.new_universe(cells_per_side, seed, unevenness=0.0,
**start) (every volume alike but for the jitter inside its boxes; `start` passes other
start amounts on, such as the fitted ones, never an unevenness) and gives each energy volume named in
`packed`, {(i, j, k) lattice place, all three even: factor}, factor times the start's
mean volume quanta, spread evenly over its own boxes (State.fill). Lengths, packings
and clock rates follow at once (cell_laws) and the books start from this state
(state.start). start_quanta and the size law's constants stay the even start's, so a
lump is still measured against an even volume. Extra energy in a volume of the grid,
never a ball in empty space: there is none. Raises ValueError for a place that is not
an energy volume. Placed starts break "nothing placed by hand" on purpose: each tests
one mechanism.
"""
from __future__ import annotations

import numpy as np

from physics.engine import ENERGY_VOLUME, Settings, State, cell_laws, new_universe, totals


def packed_start(cells_per_side: int, packed: dict, seed: int = 1, settings: Settings | None = None,
                 device: str = "cuda", start: dict | None = None) -> State:
    """An even universe with the named energy volumes packed to their factors."""
    universe = new_universe(cells_per_side, seed, **{**(start or {}), "unevenness": 0.0, "joint_unevenness": 0.0},
                            device=device)
    side = universe.side
    for place, factor in packed.items():
        if any(index % 2 or not 0 <= index < side for index in place):
            raise ValueError(f"{place} is not an energy volume of a {side}-place lattice")
        cell = (place[0] * side + place[1]) * side + place[2]
        universe.fill(cell, int(round(factor * universe.start_quanta[ENERGY_VOLUME])))
    cell_laws(universe, settings or Settings())
    universe.start = totals(universe)
    universe.start["size"] = float(universe.host("length").astype(np.float64).sum())
    return universe
