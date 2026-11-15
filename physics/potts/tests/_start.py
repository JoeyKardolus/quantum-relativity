"""Shared small starts for the cell engine's tests: an even checkerboard and one cell's boxes."""
from __future__ import annotations

import numpy as np

from physics.potts import new_universe


def even_universe(cells_per_side: int = 6, boxes_per_cell_side: int = 4):
    """A checkerboard with every box at its kind's mean energy (unevenness 0)."""
    return new_universe(cells_per_side, boxes_per_cell_side, seed=1, unevenness=0.0)


def cell_id(cells_per_side: int, cx: int, cy: int, cz: int) -> int:
    """The id new_universe gives cell (cx, cy, cz)."""
    return (cx * cells_per_side + cy) * cells_per_side + cz


def connected_everywhere(owner: np.ndarray, cells: np.ndarray) -> bool:
    """Whether every listed cell's boxes form one face-connected piece."""
    from scipy import ndimage
    return all(ndimage.label(owner == cell)[1] == 1 for cell in cells)


def table_matches_boxes(universe) -> bool:
    """Whether the GPU contact table holds exactly the pairs and face counts recounted from the boxes."""
    from physics.potts import contact_pairs, table_pairs
    recount = contact_pairs(universe.owner.numpy())
    held = table_pairs(universe)
    return all(np.array_equal(first, second) for first, second in zip(recount, held))
