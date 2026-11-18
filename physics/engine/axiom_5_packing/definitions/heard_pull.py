"""Heard pull: what one cell hears of the pull from its own neighbours, and how loud they are.

Contract: `heard_pull(cell, kind, cell_quanta, adj_at, adj_n, adj_id, pull, length,
light_speed, step_time)` (Warp func) returns a vec2d for one cell:
  [0]  the mean over its neighbour entries of `wave_weight * (neighbour's pull - own pull)`
  [1]  the mean over them of `wave_weight`
Only entries that name a living cell holding energy count: an id that is no cell is not
there, and an empty cell is empty space, endlessly long, which carries nothing. Both are
0 for a cell that hears nobody. It reads the relation tables; no neighbour is worked out
by arithmetic.
"""
from __future__ import annotations

import warp as wp

from physics.engine.axiom_3_structure.definitions.kinds import NO_CELL
from physics.engine.axiom_3_structure.definitions.neighbour_of import neighbour_count, neighbour_id
from physics.engine.axiom_5_packing.definitions.pull_wave_law import wave_weight

wp.set_module_options({"fuse_fp": False})

_NO_CELL = wp.constant(wp.int32(NO_CELL))


@wp.func
def heard_pull(cell: wp.int32, kind: wp.array(dtype=wp.int32), cell_quanta: wp.array(dtype=wp.int64),
               adj_at: wp.array(dtype=wp.int32), adj_n: wp.array(dtype=wp.int32), adj_id: wp.array(dtype=wp.int32),
               pull: wp.array(dtype=wp.float64), length: wp.array(dtype=wp.float32), light_speed: wp.float64,
               step_time: wp.float64) -> wp.vec2d:
    """The mean weighted pull difference to a cell's neighbours, and their mean weight."""
    own = pull[cell]
    own_length = wp.float64(length[cell])
    difference_sum = wp.float64(0.0)
    weight_sum = wp.float64(0.0)
    heard = int(0)
    for index in range(neighbour_count(adj_n, cell)):
        other = neighbour_id(adj_at, adj_id, cell, index)
        if kind[other] != _NO_CELL and cell_quanta[other] > wp.int64(0):
            weight = wave_weight(light_speed, step_time, own_length, wp.float64(length[other]))
            difference_sum += weight * (pull[other] - own)
            weight_sum += weight
            heard += 1
    if heard == 0:
        return wp.vec2d(wp.float64(0.0), wp.float64(0.0))
    return wp.vec2d(difference_sum / wp.float64(heard), weight_sum / wp.float64(heard))
