"""Store: where every cell is kept, built so a billion cells fit one graphics card.

A calm cell is a rule, not boxes: its whole signed total, how its swirl is laid
(`axis * 2`, plus 1 for the other sense) and its heat count, 9 bytes. Only a cell in the
own set holds boxes, in a slot; slots in use are always `0 .. count - 1`. Two plain
arrays find the slot.
"""
from dataclasses import dataclass

import warp as wp

TILE = 512  # cells in a tile: 512 cells that follow each other


@dataclass
class Store:
    across: int             # boxes along a cell's edge
    seed: int
    turn_per_step: float    # how far a calm swirl's phase turns in one step
    cell_quanta: wp.array   # int32 (cells)
    cell_laid: wp.array     # uint8 (cells)
    cell_heat: wp.array     # int32 (cells): quanta that came in with their direction taken
    tile_home: wp.array     # int32 (cells / 512): the tile's row in tile_slots, -1: all calm
    tile_slots: wp.array    # int32 (tile rows * 512): the cell's slot, -1: calm
    own_cell: wp.array      # int32 (cap): which cell holds this slot
    box_quanta: wp.array    # int32 (cap, boxes)
    box_phase: wp.array     # float32 (cap, boxes)
    pattern: wp.array       # vec2d (6, boxes): each box's share and phase in the six swirls
    step: int = 0
    count: int = 0          # slots in use
    open_tiles: int = 0     # tile rows in use
    ledger: int = 0         # quanta that changed box when cells went back to calm


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
