"""Store: where every cell is kept, built so a billion cells fit one graphics card.

A calm cell is a rule, not boxes: its whole signed total (the sign is the way its whirl
turns) and what it has made, 16 bytes. Only a cell
in the own set holds boxes, in a slot; slots in use are always `0 .. count - 1`. Two plain
arrays find the slot. With `side` above 0 a cell's number is its place (`places.py`); a
place that holds plus quanta and no slot is leaked energy gathering, not yet a cell.
"""
from dataclasses import dataclass

import warp as wp

TILE = 512  # cells in a tile: 512 cells that follow each other


@dataclass
class Store:
    across: int             # boxes along a cell's edge
    turn_per_step: float    # how far a calm swirl's phase turns in one step
    cell_quanta: wp.array   # int64 (cells): no squeeze and no merge can run it over; a box's stays int32
    cell_made: wp.array     # int64 (cells): quanta a joint has made out of itself
    cell_pull: wp.array     # float32 (cells): the pull there, handed on from cell to cell
    cell_flow: wp.array     # int64 (cells, 3): the fluid's common flow a direction, quanta times calm lengths a turn
    tile_home: wp.array     # int32 (cells / 512): the tile's row in tile_slots, -1: all calm
    tile_slots: wp.array    # int32 (tile rows * 512): the cell's slot, -1: calm
    own_cell: wp.array      # int32 (cap): which cell holds this slot
    own_squeeze: wp.array   # vec4d (cap): a joint's pull, push, volumes' distance, axis
    own_turned: wp.array    # float32 (cap): how far the cell has turned past the calm turn
    box_quanta: wp.array    # int32 (cap, boxes)
    box_phase: wp.array     # float32 (cap, boxes)
    pattern: wp.array       # vec2d (2, boxes): each box's share and phase, a volume's and a joint's
    side: int = 0           # places along a direction; 0: the cells keep no places
    leaks: bool = False     # whether the volumes leak: only in a universe that grows
    step: int = 0
    count: int = 0          # slots in use
    open_tiles: int = 0     # tile rows in use
    ledger: int = 0         # quanta that changed box when cells went back to calm
    collapses: int = 0      # joints that have collapsed
    flipped: int = 0        # minus quanta that collapses flipped to plus


@wp.func
def slot_at(tile_home: wp.array(dtype=wp.int32), tile_slots: wp.array(dtype=wp.int32),
            cell: wp.int32) -> wp.int32:
    """On the card: the slot that holds the cell's boxes, or -1 when the cell is calm."""
    row = tile_home[cell / TILE]
    if row < 0:
        return -1
    return tile_slots[row * TILE + cell % TILE]


def put(array: wp.array, index: int, value: int) -> None:
    """Write one number of a device array from the host."""
    array[index:index + 1].fill_(value)


def peek(array: wp.array, index: int) -> int:
    """Read one number of a device array on the host."""
    return int(array[index:index + 1].numpy()[0])


def entry_of(store: Store, cell: int) -> int:
    """Where `tile_slots` keeps the cell's slot, or -1 when its tile has no row yet."""
    row = peek(store.tile_home, cell // TILE)
    return -1 if row < 0 else row * TILE + cell % TILE


def slot_of(store: Store, cell: int) -> int:
    """The slot that holds the cell's boxes, or -1 when the cell is calm."""
    entry = entry_of(store, cell)
    return -1 if entry < 0 else peek(store.tile_slots, entry)
