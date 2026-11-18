"""Heard pull: what one cell hears of the pull from its own neighbours, and the room it stands in.

Contract: `heard_pull(cell, kind, cell_quanta, adj_at, adj_n, adj_id, pull, length)`
(Warp func) returns a vec2d for one cell:
  [0]  the mean over its neighbour entries of `(distance / reach) * (neighbour's pull -
       own pull)`, `distance` half the cell's length plus half the neighbour's
  [1]  `reach`: the mean of those distances
Only entries that name a living cell holding energy count: an id that is no cell is not
there, and an empty cell is empty space, endlessly long, which carries nothing. Both are
0 for a cell that hears nobody. It reads the relation tables; no neighbour is worked out
by arithmetic.
"""
from __future__ import annotations

import warp as wp

from physics.engine.axiom_3_structure.definitions.kinds import NO_CELL
from physics.engine.axiom_3_structure.definitions.neighbour_of import neighbour_count, neighbour_id

wp.set_module_options({"fuse_fp": False})

_NO_CELL = wp.constant(wp.int32(NO_CELL))
_HALF = wp.constant(wp.float64(0.5))


@wp.func
def heard_pull(cell: wp.int32, kind: wp.array(dtype=wp.int32), cell_quanta: wp.array(dtype=wp.int64),
               adj_at: wp.array(dtype=wp.int32), adj_n: wp.array(dtype=wp.int32), adj_id: wp.array(dtype=wp.int32),
               pull: wp.array(dtype=wp.float64), length: wp.array(dtype=wp.float32)) -> wp.vec2d:
    """The distance-weighted mean pull difference to a cell's neighbours, and its mean distance to them."""
    own = pull[cell]
    own_length = wp.float64(length[cell])
    difference_sum = wp.float64(0.0)
    distance_sum = wp.float64(0.0)
    heard = int(0)
    for index in range(neighbour_count(adj_n, cell)):
        other = neighbour_id(adj_at, adj_id, cell, index)
        if kind[other] != _NO_CELL and cell_quanta[other] > wp.int64(0):
            distance = _HALF * (own_length + wp.float64(length[other]))
            difference_sum += distance * (pull[other] - own)
            distance_sum += distance
            heard += 1
    if heard == 0 or not (distance_sum > wp.float64(0.0)) or wp.isinf(distance_sum):
        return wp.vec2d(wp.float64(0.0), wp.float64(0.0))
    return wp.vec2d(difference_sum / distance_sum, distance_sum / wp.float64(heard))
