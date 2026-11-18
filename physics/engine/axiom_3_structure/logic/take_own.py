"""Take own: a calm cell gets its own boxes, exactly.

The cell takes the next free slot. Each box gets its whole share of the total rounded
down, then the left-over quanta go one each from box 0 up, so the boxes sum to the
total to the quantum. The phase is the pattern's, turned on by the steps so far.
`lay` deals any number of cells in one launch; `take_own` is one cell from the host.
"""
import warp as wp

from physics.engine.axiom_2_swirls.definitions.swirl_pattern import dealt
from physics.engine.axiom_3_structure.definitions.store import TILE, Store, entry_of, peek, put

wp.set_module_options({"fuse_fp": False})


@wp.kernel
def _lay(own_cell: wp.array(dtype=wp.int32), first: wp.int32, turned: wp.float64,
         cell_quanta: wp.array(dtype=wp.int32), cell_laid: wp.array(dtype=wp.uint8),
         pattern: wp.array2d(dtype=wp.vec2d),
         box_quanta: wp.array2d(dtype=wp.int32), box_phase: wp.array2d(dtype=wp.float32)):
    slot = first + wp.tid()
    sign = wp.sign(cell_quanta[own_cell[slot]])  # a joint is dealt like a volume, in minus
    total = wp.abs(cell_quanta[own_cell[slot]])
    laid = wp.int32(cell_laid[own_cell[slot]])
    boxes = pattern.shape[1]
    left_over = total
    for box in range(boxes):
        left_over -= dealt(total, pattern[laid, box][0], boxes)
    for box in range(boxes):
        angle = pattern[laid, box][1] + turned
        angle -= wp.float64(wp.tau) * wp.floor((angle + wp.float64(wp.pi)) / wp.float64(wp.tau))
        box_phase[slot, box] = wp.float32(angle)
        share = dealt(total, pattern[laid, box][0], boxes)
        if box < left_over:
            share += 1
        box_quanta[slot, box] = sign * share


def lay(store: Store, own_cell: wp.array, first: int, number: int,
        box_quanta: wp.array, box_phase: wp.array) -> None:
    """Deal the rule's boxes of `number` cells, named by `own_cell` from `first` on."""
    wp.launch(_lay, dim=number, device=box_quanta.device,
              inputs=[own_cell, first, wp.float64(store.turn_per_step * store.step),
                      store.cell_quanta, store.cell_laid, store.pattern, box_quanta, box_phase])


def take_own(store: Store, cell: int) -> None:
    """Give a calm cell its own boxes; an own cell is left as it is."""
    entry = entry_of(store, cell)
    if entry >= 0 and peek(store.tile_slots, entry) >= 0:
        return
    if store.count == store.own_cell.shape[0]:
        raise RuntimeError(f"the own set is full: {store.count} cells hold their own boxes")
    if entry < 0 and store.open_tiles * TILE == store.tile_slots.shape[0]:
        raise RuntimeError(f"the tile rows are full: {store.open_tiles} tiles hold own cells")
    if entry < 0:
        put(store.tile_home, cell // TILE, store.open_tiles)
        store.open_tiles += 1
    put(store.tile_slots, entry_of(store, cell), store.count)
    put(store.own_cell, store.count, cell)
    lay(store, store.own_cell, store.count, 1, store.box_quanta, store.box_phase)
    store.count += 1
