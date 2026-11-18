"""Packed start: an even grid in which chosen energy volumes hold more energy (a placed start).

Contract: `packed_start(cells_per_side, packed, seed=1, settings=None, device="cuda",
start=None)` builds physics.engine.new_universe with the settings (default Settings();
`start` puts a candidate's start amounts in, start_settings) made even: `unevenness`
and `joint_unevenness` 0, so every volume is alike but for the jitter inside its boxes.
Each energy volume named in `packed`, {(i, j, k) lattice place, all three even: factor},
then holds round(factor * even_volume_quanta) quanta spread evenly over its own boxes
(even_fill); its phases are left as its swirl wound them. Every cell's count, length,
packing and clock rate follow at once by the engine's own laws (count_cells, size_cells,
pack_cells) and the books start from this state (`start_books`). The pull is left as the
even grid's resting pull: the lump's pull spreads from the first step on. Extra energy
in a volume of the grid, never a ball in empty space. Returns the Grid. Raises
ValueError for a place that is not an energy volume or a factor that is not above 0.
"""
from __future__ import annotations

from dataclasses import replace

from physics.engine import Grid, Settings, new_universe, read_books
from physics.engine.axiom_1_energy.logic.count_cells import count_cells
from physics.engine.axiom_3_structure.logic.size_cells import size_cells
from physics.engine.axiom_5_packing.logic.pack_cells import pack_cells
from physics.scenes.placed_start.definitions.even_fill import even_fill
from physics.scenes.placed_start.definitions.place_cell import place_cell
from physics.scenes.run.definitions.even_volume_quanta import even_volume_quanta
from physics.scenes.run.definitions.start_settings import start_settings


def packed_start(cells_per_side: int, packed: dict, seed: int = 1, settings: Settings | None = None,
                 device: str = "cuda", start: dict | None = None) -> Grid:
    """An even universe with the named energy volumes packed to their factors."""
    settings = replace(start_settings(settings, start), unevenness=0.0, joint_unevenness=0.0)
    universe = new_universe(settings, cells_per_side, seed, device=device)
    even = even_volume_quanta(settings)
    for place, factor in packed.items():
        if not factor > 0.0:
            raise ValueError(f"a packed volume holds more than 0 times an even one, got {factor}")
        universe.set_block("quanta", place_cell(universe.side, place), even_fill(round(factor * even)))
    count_cells(universe)
    size_cells(universe, settings)
    pack_cells(universe, settings)
    universe.start_books = read_books(universe)
    return universe
