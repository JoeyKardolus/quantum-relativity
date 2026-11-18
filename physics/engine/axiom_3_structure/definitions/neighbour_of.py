"""Neighbour of: how a kernel reads a cell's relations, always from the table.

Contract: three Warp funcs over the relation arrays (structure_arrays.py, grid.py). A
cell's neighbour entries are rows `adj_at[cell]` to `adj_at[cell] + adj_n[cell] - 1` of
the pools `adj_id` and `adj_code`; no neighbour is ever worked out by arithmetic.
  neighbour_count(adj_n, cell)                  how many entries the cell has
  neighbour_id(adj_at, adj_id, cell, index)     the id of entry `index`
  neighbour_code(adj_at, adj_code, cell, index)  the direction that neighbour lies in,
                                                seen from the cell
`index` runs from 0 to the count less 1; nothing checks it.
"""
from __future__ import annotations

import warp as wp

wp.set_module_options({"fuse_fp": False})


@wp.func
def neighbour_count(adj_n: wp.array(dtype=wp.int32), cell: wp.int32) -> wp.int32:
    """How many neighbour entries a cell has."""
    return adj_n[cell]


@wp.func
def neighbour_id(adj_at: wp.array(dtype=wp.int32), adj_id: wp.array(dtype=wp.int32), cell: wp.int32,
                 index: wp.int32) -> wp.int32:
    """The id of a cell's neighbour entry number `index`."""
    return adj_id[adj_at[cell] + index]


@wp.func
def neighbour_code(adj_at: wp.array(dtype=wp.int32), adj_code: wp.array(dtype=wp.int32), cell: wp.int32,
                   index: wp.int32) -> wp.int32:
    """The direction code of a cell's neighbour entry number `index`."""
    return adj_code[adj_at[cell] + index]
