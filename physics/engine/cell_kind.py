"""Cell kind (Warp func): what a cell of the forced cell lattice is, read from its lattice index alone.

Contract: `cell_kind(cell, extent, cells)` takes a cell id of the extended lattice
(cell_place.py: ids over the batch's extent per axis) and the universe's real cells per
axis `cells` (state.counts), and returns
   1  an energy volume: all three lattice indices even and in 2 .. cells[axis] - 1
  -1  a joint: every other real cell (indices 1 .. cells[axis])
   0  the ring: empty space round the lattice (an index 0 or past cells[axis])
So every energy volume has joints on all 26 sides, the joints form one web, two
face-neighbouring volumes share exactly one joint (the cell halfway between them) and
the outer layer of real cells is joints (owner, 2026-11-16: the cells form a forced
grid, neighbours fixed by index, the same rule for every cell). A box's quanta carry
its cell's sign: plus in a volume, minus in a joint, none in the ring. A volume split
(volume_split.py) adds lattice planes, so `cells` grows along one axis.
"""
from __future__ import annotations

import warp as wp

from physics.engine.cell_place import cell_place


@wp.func
def cell_kind(cell: int, extent: wp.vec3i, cells: wp.vec3i) -> int:
    """1 energy volume, -1 joint, 0 ring (empty space) for an extended-lattice cell id."""
    place = cell_place(cell, extent)
    kind = int(1)
    for axis in range(3):
        index = place[axis]
        if index <= 0 or index > cells[axis]:
            return 0
        if index % 2 == 1 or index < 2 or index > cells[axis] - 1:
            kind = -1
    return kind
