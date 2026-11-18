"""Lay lattice: a whole start lattice of relations is written into an empty grid at once.

Contract: `lay_lattice(grid, neighbour_ids, neighbour_codes, ways)` takes 26 neighbour
entries a cell and every way of a start lattice and writes them into the grid's host
tables and onto the device, every way in pass 0. `neighbour_ids` is (count, 26);
`neighbour_codes` is (count, 26) or the 26 codes every cell uses; `ways` is (number, 5)
rows `(a, b, mouth_a, mouth_b, joint)`. `start_grid` calls it once; nothing calls it
after the start.
"""
from __future__ import annotations

import numpy as np

from physics.engine.axiom_3_structure.definitions.grid import JOINT_COLUMN, PASS_COLUMN, START_ENTRIES, WAY_COLUMNS


def lay_lattice(grid, neighbour_ids, neighbour_codes, ways) -> None:
    """Take the whole start lattice at once: 26 entries a cell and every way, all in pass 0.

    `neighbour_ids` is (count, 26); `neighbour_codes` is (count, 26) or the 26 codes
    every cell uses; `ways` is (number, 5) rows `(a, b, mouth_a, mouth_b, joint)`.
    """
    ids = np.ascontiguousarray(neighbour_ids, dtype=np.int32)
    if ids.shape != (grid.count, START_ENTRIES):
        raise ValueError(f"neighbour_ids must be {(grid.count, START_ENTRIES)}, got {ids.shape}")
    codes = np.ascontiguousarray(np.broadcast_to(np.asarray(neighbour_codes, dtype=np.int32), ids.shape))
    rows = np.asarray(ways, dtype=np.int32).reshape(-1, 5)
    rows = rows[np.lexsort((rows[:, 2], rows[:, 1], rows[:, 0], rows[:, 4]))]
    grid.adj_used = ids.size
    grid.room_in_pool(0)
    grid.pool_id[:ids.size], grid.pool_code[:ids.size] = ids.ravel(), codes.ravel()
    grid.adj_start[:grid.count] = np.arange(grid.count, dtype=np.int32) * START_ENTRIES
    grid.adj_count[:grid.count] = grid.adj_room[:grid.count] = START_ENTRIES
    grid.way_count = 0
    grid.grow(0, len(rows))
    grid.way_count = len(rows)
    grid.way_table[:] = -1
    grid.way_table[:len(rows), :JOINT_COLUMN + 1] = rows
    grid.way_table[:len(rows), PASS_COLUMN] = 0
    grid.way_number[:] = 0
    grid.way_number[:grid.count] = np.bincount(rows[:, 4], minlength=grid.count)[:grid.count]
    grid.way_first[:grid.count] = np.cumsum(grid.way_number[:grid.count]) - grid.way_number[:grid.count]
    grid.joint_slots, grid.free_slots, grid.unsent = {}, [], set()
    grid.way_passes = 1
    grid.send_pools()
    grid.adj_at.assign(grid.adj_start)
    grid.adj_n.assign(grid.adj_count)
    for column, name in enumerate(WAY_COLUMNS):
        getattr(grid, name).assign(np.ascontiguousarray(grid.way_table[:, column]))
