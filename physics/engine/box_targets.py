"""Box targets: the cell whose moving region covers each Planck box's centre, looked for among its owner's neighbours.

Contract: `box_targets(state)` writes state.target for every box of a state with a cell
lattice (state.cells_per_side > 0; without one it returns at once): the box's own cell
(state.owner) while the box's centre (x + 0.5, y + 0.5, z + 0.5) lies in it, else the
cell, of the owner's 26 lattice neighbours, that the centre lies deepest in
(cell_score.py; the ring of empty space counts, so a box can fall out of the lattice
and an empty box can be taken in). Corners move less than a box a step
(move_corners.py), so the covering cell is always the owner or one of its neighbours;
where a cell is thinner than a box the box steps across it over two steps. Neighbours
are fixed by lattice index: no label is read off the boxes. The box changes owner only
later, once its quanta have flowed out (drain.py, hand_over.py). Returns None.
"""
from __future__ import annotations

import warp as wp

from physics.engine.cell_place import cell_place
from physics.engine.cell_score import cell_score
from physics.engine.state import State


@wp.kernel
def _target(owner: wp.array3d(dtype=wp.int32), corners: wp.array3d(dtype=wp.vec3), side: int,
            cells_per_side: int, target: wp.array3d(dtype=wp.int32)):
    ti, tj, tk = wp.tid()
    home = ti // side
    own = owner[ti, tj, tk]
    point = wp.vec3(float(ti - home * side) + 0.5, float(tj) + 0.5, float(tk) + 0.5)
    best = cell_score(corners, home, own, cells_per_side, point)
    chosen = own
    if best < 0.0:
        place = cell_place(own, cells_per_side)
        extent = cells_per_side + 2
        for index in range(27):
            near = place + wp.vec3i(index // 9 - 1, index // 3 % 3 - 1, index % 3 - 1)
            if index != 13 and near[0] >= 0 and near[1] >= 0 and near[2] >= 0 and near[0] < extent \
                    and near[1] < extent and near[2] < extent:
                cell = (near[0] * extent + near[1]) * extent + near[2]
                score = cell_score(corners, home, cell, cells_per_side, point)
                if score > best:
                    best = score
                    chosen = cell
    target[ti, tj, tk] = chosen


def box_targets(state: State) -> None:
    """Write each box's covering cell into state.target."""
    if state.cells_per_side == 0:
        return
    wp.launch(_target, dim=state.shape, device=state.quanta.device,
              inputs=[state.owner, state.corners, state.side, state.cells_per_side, state.target])
