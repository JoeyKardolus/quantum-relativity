"""Brick start: energy volumes as blocks of plus boxes, each wrapped in joints of minus boxes, inside an empty margin.

Contract: `brick_start(cells, cell_boxes, margin, volume_quanta=1000, joint_quanta=1000,
unevenness=0.1, seed=1)` returns (quanta, phase) host arrays (int32 and float32) of side
cells * cell_boxes + 2 * margin. The filled cube is cells**3 cells of cell_boxes**3
boxes; a cell whose three cell coordinates are all odd and not on the outer layer is an
energy volume (every box +q, q = volume_quanta * (1 + unevenness * x), x uniform in
[-1, 1) per volume from `seed`), every other cell is joint (every box -joint_quanta), so
each energy volume is wrapped on all 26 sides and the joints form one web (bible "The
foundation"). The margin is empty (0, no space). Phase 0 everywhere: a fluid at rest.
Raises ValueError for fewer than 3 cells a side or a cell under 2 boxes.
"""
from __future__ import annotations

import numpy as np


def brick_start(cells: int, cell_boxes: int, margin: int, volume_quanta: int = 1000, joint_quanta: int = 1000,
                unevenness: float = 0.1, seed: int = 1) -> tuple:
    """Signed quanta and phases of a wrapped brick grid of energy volumes and joints in an empty margin."""
    if cells < 3 or cell_boxes < 2:
        raise ValueError("need at least 3 cells a side of at least 2 boxes")
    rng = np.random.default_rng(seed)
    cell_quanta = np.full((cells,) * 3, -int(joint_quanta), dtype=np.int64)
    odd = (np.arange(cells) % 2 == 1) & (np.arange(cells) < cells - 1)
    volume = odd[:, None, None] & odd[None, :, None] & odd[None, None, :]
    factors = 1.0 + unevenness * rng.uniform(-1.0, 1.0, size=int(volume.sum()))
    cell_quanta[volume] = np.rint(volume_quanta * factors).astype(np.int64)
    filled = np.repeat(np.repeat(np.repeat(cell_quanta, cell_boxes, 0), cell_boxes, 1), cell_boxes, 2)
    side = cells * cell_boxes + 2 * margin
    quanta = np.zeros((side,) * 3, dtype=np.int32)
    quanta[margin:side - margin, margin:side - margin, margin:side - margin] = filled
    return quanta, np.zeros((side,) * 3, dtype=np.float32)
