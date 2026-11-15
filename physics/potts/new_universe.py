"""New universe: energy volumes wrapped in a web of joints, cells of Planck boxes, uneven energy, in a margin.

Contract: `new_universe(cells_per_side, boxes_per_cell_side=6, seed=1, unevenness=0.3,
slope=-3, margin=None, device="cuda")` returns a Universe (universe.py) at rest.
The filled cube is cells_per_side**3 cells, each a block of k**3 boxes (k =
boxes_per_cell_side); cell (cx, cy, cz) is an energy volume (A) when cx, cy and cz are
all odd and none is n - 1 (the outer layer), else a joint (B). So every energy volume
is fully wrapped in joints (bible foundation 1, owner 2026-11-16): no box of another
energy volume, and no empty box, lies among the 26 neighbours (face, edge, corner) of
any of its boxes, since two energy volumes are at least one joint block (k >= 2 boxes)
apart. The joints form one connected web, like mortar between bricks: a face joint (one
coordinate even) lies between two energy volumes, touching each with a big side, and
meets the edge joints (two even) round its rim, which meet the corner joints (three
even); where joints meet are the seams (seam_box.py). Every cell, either kind, holds
k**3 boxes (rule 8: a joint is a full cell). Cell ids run cx-major, (cx * n + cy) * n + cz.
Around it lies an empty margin of `margin` boxes on every side (default: the filled
cube's own width, so space can grow to three times its width before a wall), rounded
up so the grid side is a multiple of 8 (multigrid levels), that the cells claim as
space grows.
Energy is uneven with ripples at every size: with g the uneven field over the filled
boxes (uneven_field.py: seed, slope, rms unevenness), every box, energy volume and
joint alike, holds exp(g) scaled so the mean box holds exactly 1, as the energy
fluid's psi = sqrt(exp(g)) with phase 0: at rest. Joints are full cells
(bible rule 8, owner 2026-11-15): a joint's size follows its own energy by the same
law as an energy volume's (cell_sums.py, one cell_rest for both kinds), so with the
same energy per box both kinds start at comparable sizes: each cell's want differs
from k**3 only by its own share of the ripples. The start's size correction
(size_correction.py, run by the first step) then lets each size follow its energy.
Raises ValueError for cells_per_side < 3 (no energy volume fits), k < 2, a negative margin and the uneven
field's errors (filled side under 8).
"""
from __future__ import annotations

import numpy as np

from physics.potts.uneven_field import uneven_field
from physics.potts.universe import ENERGY_VOLUME, JOINT, Universe


def new_universe(cells_per_side: int, boxes_per_cell_side: int = 6, seed: int = 1, unevenness: float = 0.3,
                 slope: float = -3.0, margin: int | None = None, device: str = "cuda") -> Universe:
    """Seeded energy volumes wrapped in a joint web, uneven energy, at rest, inside an empty margin."""
    if cells_per_side < 3 or boxes_per_cell_side < 2:
        raise ValueError(f"need at least 3 cells per side of at least 2 boxes, got {cells_per_side}, "
                         f"{boxes_per_cell_side}")
    filled = cells_per_side * boxes_per_cell_side
    margin = filled if margin is None else int(margin)
    if margin < 0:
        raise ValueError(f"margin must be >= 0, got {margin}")
    side = -(-(filled + 2 * margin) // 8) * 8
    low = (side - filled) // 2
    cell_axis = np.arange(filled) // boxes_per_cell_side
    cx, cy, cz = np.meshgrid(cell_axis, cell_axis, cell_axis, indexing="ij")
    cell_id = (cx * cells_per_side + cy) * cells_per_side + cz
    cell_index = np.arange(cells_per_side)
    gx, gy, gz = np.meshgrid(cell_index, cell_index, cell_index, indexing="ij")
    inner_odd = lambda index: (index % 2 == 1) & (index < cells_per_side - 1)
    kind = np.where(inner_odd(gx) & inner_odd(gy) & inner_odd(gz), ENERGY_VOLUME, JOINT).ravel().astype(np.int32)
    raw = np.exp(uneven_field(filled, seed, slope, unevenness))
    raw /= raw.mean()
    owner = np.full((side,) * 3, -1, dtype=np.int32)
    energy = np.zeros((side,) * 3, dtype=np.float64)
    inside = (slice(low, low + filled),) * 3
    owner[inside] = cell_id
    energy[inside] = raw
    return Universe.from_host(owner, energy, kind, boxes_per_cell_side, seed, device)
