"""Structure arrays: the state axiom 3 owns, the cells' kinds, sizes and relations.

Contract: `ARRAYS = {name: (dtype, shape, each_step)}` and `HOST = {name: default}` in the
form every axiom's `*_arrays.py` uses (grid.py reads them).
Per cell id (GPU):
  kind      int32: 1 energy volume, -1 joint, 0 no cell (kinds.py)
  length    float32: the cell's size by the size law, written only by `size_cells`;
            0 for an id that is no cell
  adj_at    int32: where the cell's neighbour entries start in the pools `adj_id` and
            `adj_code` (the pools are owned and sized by `Grid`, not listed here)
  adj_n     int32: how many neighbour entries it has (26 at the start)
Per way slot (GPU), one row per way; a way keeps its slot for as long as it lives:
  way_a, way_b            int32: the two energy volumes, a the lower id
  way_mouth_a, way_mouth_b  int32: the code of the joint as seen from a, and from b
  way_joint               int32: the joint the way goes through; -1: no way in this slot
  way_pass                int32: ways that share a mouth of a volume have different passes
Host values:
  version     counts tears and collapses
  way_passes  how many passes the ways need (1 at the start)
  changed     this step's events, a list of records
  fell_into   per cell: the volume a vanished volume fell into, -1 for none
  torn_from   per cell: the volume a tear made this cell from, -1 for none
  made_kind   per cell: the kind the cell was made with (0: never made)
A per-cell host array is written `(dtype, "cell")`, or `(dtype, "cell", fill)` when its
empty value is not 0.
"""
from __future__ import annotations

ARRAYS = {
    "kind": ("int32", "cell", False),
    "length": ("float32", "cell", False),
    "adj_at": ("int32", "cell", False),
    "adj_n": ("int32", "cell", False),
    "way_a": ("int32", "way", False),
    "way_b": ("int32", "way", False),
    "way_mouth_a": ("int32", "way", False),
    "way_mouth_b": ("int32", "way", False),
    "way_joint": ("int32", "way", False),
    "way_pass": ("int32", "way", False),
}

HOST = {
    "version": 0,
    "way_passes": 1,
    "changed": [],
    "fell_into": ("int32", "cell", -1),
    "torn_from": ("int32", "cell", -1),
    "made_kind": ("int32", "cell"),
}
