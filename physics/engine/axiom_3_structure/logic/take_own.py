"""Take own: a calm cell gets its own boxes, exactly.

The cell takes the next free slot. Each box gets its whole share of the total rounded
down, then the left-over quanta go one each from box 0 up, so the boxes sum to the
total to the quantum. The phase is the pattern's, turned on by the steps so far at the
calm turn; what its quanta add starts from here (`turn.py`). A minus total is laid as the mirror.
`lay` deals any number of cells in one launch; `take_all` takes many from the host.
"""
import warp as wp

from physics.engine.axiom_2_swirls.definitions.swirl_pattern import dealt
from physics.engine.axiom_3_structure.definitions.store import TILE, Store, slot_of

wp.set_module_options({"fuse_fp": False})


@wp.kernel
def _lay(own_cell: wp.array(dtype=wp.int32), first: wp.int32, turned: wp.float64,
         cell_quanta: wp.array(dtype=wp.int64),
         pattern: wp.array2d(dtype=wp.vec2d),
         box_quanta: wp.array2d(dtype=wp.int32), box_phase: wp.array2d(dtype=wp.float32)):
    slot = first + wp.tid()
    total = wp.abs(cell_quanta[own_cell[slot]])
    laid = wp.int32(cell_quanta[own_cell[slot]] < wp.int64(0))  # a joint's whirl is a volume's in a mirror
    sign = 1 - 2 * laid  # a joint is dealt like a volume, in minus
    boxes = pattern.shape[1]
    left_over = total
    for box in range(boxes):
        left_over -= wp.int64(dealt(total, pattern[laid, box][0], boxes))
    for box in range(boxes):
        angle = pattern[laid, box][1] + turned
        angle -= wp.float64(wp.tau) * wp.floor((angle + wp.float64(wp.pi)) / wp.float64(wp.tau))
        box_phase[slot, box] = wp.float32(angle)
        share = dealt(total, pattern[laid, box][0], boxes)
        if wp.int64(box) < left_over:
            share += 1
        box_quanta[slot, box] = sign * share


def lay(store: Store, own_cell: wp.array, first: int, number: int,
        box_quanta: wp.array, box_phase: wp.array) -> None:
    """Deal the rule's boxes of `number` cells, named by `own_cell` from `first` on."""
    wp.launch(_lay, dim=number, device=box_quanta.device,
              inputs=[own_cell, first, wp.float64(store.turn_per_step * store.step),
                      store.cell_quanta, store.pattern, box_quanta, box_phase])
    if box_phase is store.box_phase:
        store.own_turned[first:first + number].zero_()  # laid fresh: nothing turned past calm yet


def take_all(store: Store, cells: list) -> None:
    """Give calm cells their own boxes, the next free slots in the order given, in one launch."""
    home, slots, own = (kept.numpy() for kept in (store.tile_home, store.tile_slots, store.own_cell))
    if store.count + len(cells) > len(own):
        raise RuntimeError(f"the own set is full: {store.count} cells hold their own boxes")
    for slot, cell in enumerate(cells, store.count):
        if home[cell // TILE] < 0 and store.open_tiles * TILE == len(slots):
            raise RuntimeError(f"the tile rows are full: {store.open_tiles} tiles hold own cells")
        if home[cell // TILE] < 0:
            home[cell // TILE], store.open_tiles = store.open_tiles, store.open_tiles + 1
        slots[home[cell // TILE] * TILE + cell % TILE], own[slot] = slot, cell
    for kept, now in ((store.tile_home, home), (store.tile_slots, slots), (store.own_cell, own)):
        kept.assign(now)
    lay(store, store.own_cell, store.count, len(cells), store.box_quanta, store.box_phase)
    store.count += len(cells)


def take_own(store: Store, cell: int) -> None:
    """Give a calm cell its own boxes; an own cell is left as it is."""
    if slot_of(store, cell) < 0:
        take_all(store, [cell])
