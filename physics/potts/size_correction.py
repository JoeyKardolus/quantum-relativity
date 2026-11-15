"""Size correction: the start's one correction that lets each cell's size follow its energy (foundation 1).

Contract: `size_correction(universe, settings)` runs sweeps of box moves in correcting
mode (box_moves.py: only the size term, aimed at each cell's want, no clock, no move
limit), refreshing the cell sums and the active region after each, until a sweep moves
under a thousandth of the occupied boxes or MOST_SWEEPS (64) have run. The fluid does
not run, so no energy flows; boxes carry their energy as every move does, and total
energy stays exact. Every cell's previous_boxes is then set to its size, so the joints'
swing starts at rest. Sets universe.corrected and returns the sweeps it ran. The first
step() calls it once; after it the rules alone bend, squeeze and stretch the grid.
"""
from __future__ import annotations

import warp as wp

from physics.potts.active_region import active_region
from physics.potts.box_moves import box_moves
from physics.potts.cell_sums import cell_sums
from physics.potts.settings import Settings
from physics.potts.universe import Universe

MOST_SWEEPS = 64


def size_correction(universe: Universe, settings: Settings) -> int:
    """Move boxes until every cell's size follows its energy; return the sweeps taken."""
    sweeps = 0
    cell_sums(universe, settings)
    while sweeps < MOST_SWEEPS:
        occupied = int(universe.cell_boxes.numpy().sum())
        moved = box_moves(universe, settings, correcting=True)
        sweeps += 1
        active_region(universe)
        cell_sums(universe, settings)
        if moved < 0.001 * occupied:
            break
    wp.copy(universe.previous_boxes, wp.array(universe.cell_boxes.numpy().astype("float32"), dtype=wp.float32,
                                              device=universe.owner.device))
    universe.corrected = True
    return sweeps
