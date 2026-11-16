"""Cell kind (Warp func): what a cell of the forced cell lattice is, read from its lattice index alone.

Contract: `cell_kind(cell, cells_per_side)` takes a cell id of the extended lattice
(cell_place.py: cells_per_side real cells a side plus one ring cell on each end, ids
(a * m + b) * m + c with m = cells_per_side + 2) and returns
   1  an energy volume: all three lattice indices even and in 2 .. cells_per_side - 1
  -1  a joint: every other real cell (indices 1 .. cells_per_side)
   0  the ring: empty space round the lattice (an index 0 or cells_per_side + 1)
So every energy volume has joints on all 26 sides, the joints form one web, two
face-neighbouring volumes share exactly one joint (the cell halfway between them) and
the outer layer of real cells is joints (owner, 2026-11-16: the cells form a forced
grid, neighbours fixed by index, the same rule for every cell). A box's quanta carry
its cell's sign: plus in a volume, minus in a joint, none in the ring.
"""
from __future__ import annotations

import warp as wp

from physics.engine.cell_place import cell_place


@wp.func
def cell_kind(cell: int, cells_per_side: int) -> int:
    """1 energy volume, -1 joint, 0 ring (empty space) for an extended-lattice cell id."""
    place = cell_place(cell, cells_per_side)
    last = cells_per_side + 1
    kind = int(1)
    for axis in range(3):
        index = place[axis]
        if index <= 0 or index >= last:
            return 0
        if index % 2 == 1 or index < 2 or index > cells_per_side - 1:
            kind = -1
    return kind
