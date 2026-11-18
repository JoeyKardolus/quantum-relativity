"""Orbit start: an even grid with a big packed volume and, beside it, a small one whose fluid is set moving sideways.

Contract: `orbit_start(cells_per_side, distance, big_energy, small_energy, sideways,
seed=1, settings=None, device="cuda", start=None)` builds a placed start (packed_start,
which `start` reaches): the middle energy volume holds big_energy times an even
volume's quanta, and the volume `distance` volumes from it along the first axis holds
small_energy times. The small volume's phases then get a ramp of `sideways` radians a
box along the second axis (sideways_ramp): its fluid starts moving across the line to
the big one. Returns `(universe, twin, small, middle)`: the Grid, a second Grid from the
same seed with the big volume only (no small one, no ramp: what the grid does by
itself, to be stepped beside and subtracted), the small volume's index triple on the
volumes' lattice and the big one's index along each axis. Raises ValueError for a
distance under 2 or of a quarter of the cells a side or more (half way round the box
has no side to be on).
"""
from __future__ import annotations

from physics.engine import Grid, Settings
from physics.scenes.orbit.definitions.sideways_ramp import sideways_ramp
from physics.scenes.placed_start.definitions.place_cell import place_cell
from physics.scenes.placed_start.logic.packed_start import packed_start

_RAMP_AXIS = 1


def orbit_start(cells_per_side: int, distance: int, big_energy: float, small_energy: float, sideways: float,
                seed: int = 1, settings: Settings | None = None, device: str = "cuda",
                start: dict | None = None) -> tuple[Grid, Grid, tuple, int]:
    """The universe, its twin without the small volume, the small volume's place and the middle."""
    across = cells_per_side // 2
    middle = across // 2
    if not 2 <= distance < across // 2:
        raise ValueError(f"the small volume must lie 2 .. {across // 2 - 1} volumes out, got {distance}")
    small = ((middle + distance) % across, middle, middle)
    small_place = tuple(2 * index for index in small)
    universe = packed_start(cells_per_side, {(2 * middle,) * 3: big_energy, small_place: small_energy},
                            seed, settings, device, start)
    cell = place_cell(universe.side, small_place)
    universe.set_block("phase", cell, sideways_ramp(universe.block("phase", cell), sideways, _RAMP_AXIS))
    twin = packed_start(cells_per_side, {(2 * middle,) * 3: big_energy}, seed, settings, device, start)
    return universe, twin, small, middle
