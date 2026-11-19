"""Seat: many places get their own boxes at once, on the card, whatever their number.

`take_all` does this from the host for a scene's start; a step uses this. Every place
flagged 1 takes a slot: the next free ones, in the order of the places' numbers (a running
count over the flags gives each its rank), and its boxes are dealt from its total as whole
ring whirls (`lay`). A tile that holds its first own cell opens a row the same way. Two
single numbers come back to the host: how many cells, how many rows.
"""
import warp as wp
from warp.utils import array_scan

from physics.engine.axiom_3_structure.definitions.store import TILE, Store, peek
from physics.engine.axiom_3_structure.logic.spare import spare
from physics.engine.axiom_3_structure.logic.take_own import lay


@wp.kernel
def _wanted(fresh: wp.array(dtype=wp.int32), tile_home: wp.array(dtype=wp.int32),
            opens: wp.array(dtype=wp.int32)):
    cell = wp.tid()
    if fresh[cell] == 1 and tile_home[cell / TILE] < 0:
        opens[cell / TILE] = 1  # every writer writes the same 1


@wp.kernel
def _open(opens: wp.array(dtype=wp.int32), rows: wp.array(dtype=wp.int32), first: wp.int32,
          tile_home: wp.array(dtype=wp.int32)):
    if opens[wp.tid()] == 1:
        tile_home[wp.tid()] = first + rows[wp.tid()] - 1


@wp.kernel
def _seat(fresh: wp.array(dtype=wp.int32), rank: wp.array(dtype=wp.int32), first: wp.int32,
          tile_home: wp.array(dtype=wp.int32), tile_slots: wp.array(dtype=wp.int32),
          own_cell: wp.array(dtype=wp.int32)):
    cell = wp.tid()
    if fresh[cell] == 1:
        own_cell[first + rank[cell] - 1] = cell
        tile_slots[tile_home[cell / TILE] * TILE + cell % TILE] = first + rank[cell] - 1


def seat(store: Store, fresh: wp.array) -> int:
    """Give every place flagged in `fresh` its own boxes; says how many there were."""
    device, tiles = fresh.device, store.tile_home.shape[0]
    rank, opens, rows = (spare(store, name, size, wp.int32) for name, size in
                         (("rank", fresh.shape[0]), ("opens", tiles), ("rows", tiles)))
    array_scan(fresh, rank, True)
    number = peek(rank, fresh.shape[0] - 1)
    if store.count + number > store.own_cell.shape[0]:
        raise RuntimeError(f"the own set is full: {store.count} cells hold their own boxes")
    opens.zero_()
    wp.launch(_wanted, dim=fresh.shape[0], device=device, inputs=[fresh, store.tile_home, opens])
    array_scan(opens, rows, True)
    opened = peek(rows, tiles - 1)
    if (store.open_tiles + opened) * TILE > store.tile_slots.shape[0]:
        raise RuntimeError(f"the tile rows are full: {store.open_tiles} tiles hold own cells")
    wp.launch(_open, dim=tiles, device=device, inputs=[opens, rows, store.open_tiles, store.tile_home])
    wp.launch(_seat, dim=fresh.shape[0], device=device,
              inputs=[fresh, rank, store.count, store.tile_home, store.tile_slots, store.own_cell])
    lay(store, store.own_cell, store.count, number, store.box_quanta, store.box_phase)
    store.open_tiles, store.count = store.open_tiles + opened, store.count + number
    return number
