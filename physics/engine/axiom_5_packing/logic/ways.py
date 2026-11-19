"""Ways: the way from a volume to another body through the cells' sizes, on the card.

`between.py` has the law, read on the host for checks: along one direction the way is the
sizes of the cells between in a row, counted along the row of the one and along the row of
the other, and the mean taken. `middles` lays, for every row of places along every
direction, how far each place's middle lies from the row's start: one launch, a thread a
row. `way_to` reads the way off that. The other body is a volume or a group: it stands at
`at` (a group between places: by its share of the way between the two volumes' places
either side) and its row is that of `lead`, its fullest volume. For a volume this is
`between`'s way to the last digit.
"""
import warp as wp

from physics.engine.axiom_3_structure.definitions.kinds import QUANTA_A_BOX
from physics.engine.axiom_3_structure.definitions.places import stride_of
from physics.engine.axiom_3_structure.definitions.store import Store, slot_at
from physics.engine.axiom_3_structure.logic.spare import spare

wp.set_module_options({"fuse_fp": False, "enable_backward": False})


@wp.kernel
def _middles(cell_quanta: wp.array(dtype=wp.int64), tile_home: wp.array(dtype=wp.int32),
             tile_slots: wp.array(dtype=wp.int32), side: wp.int32, calm: wp.float64,
             middle: wp.array2d(dtype=wp.float64)):
    axis, first, second = wp.tid()
    start = first * stride_of(side, (axis + 1) % 3) + second * stride_of(side, (axis + 2) % 3)
    so_far = wp.float64(0.0)
    for along in range(side):
        cell, size = start + along * stride_of(side, axis), wp.float64(0.0)
        if cell_quanta[cell] != wp.int64(0) and slot_at(tile_home, tile_slots, cell) >= 0:
            size = wp.float64(1.0) / wp.abs(wp.float64(cell_quanta[cell]) / calm)
        so_far += size
        middle[axis, cell] = so_far - size / wp.float64(2.0)


@wp.func
def plus_at(cell: wp.int32, cell_quanta: wp.array(dtype=wp.int64), tile_home: wp.array(dtype=wp.int32),
            tile_slots: wp.array(dtype=wp.int32), calm: wp.float64) -> wp.float64:
    """The plus energy of the own cell at a place, in calm volumes; 0 where none stands."""
    if cell_quanta[cell] <= wp.int64(0) or slot_at(tile_home, tile_slots, cell) < 0:
        return wp.float64(0.0)
    return wp.float64(cell_quanta[cell]) / calm


@wp.func
def in_row(middle: wp.array2d(dtype=wp.float64), side: wp.int32, axis: wp.int32, cell: wp.int32,
           to: wp.float64) -> wp.float64:
    """How far along `cell`'s row along `axis` the step `to` lies (between two volumes' places: by its share)."""
    apart = stride_of(side, axis)
    start = cell - cell / apart % side * apart
    low = wp.min(2 * wp.int32(to * wp.float64(0.5)), side - 1)
    share = (to - wp.float64(low)) * wp.float64(0.5)
    return (middle[axis, start + low * apart] * (wp.float64(1.0) - share)
            + middle[axis, start + wp.min(low + 2, side - 1) * apart] * share)


@wp.func
def way_to(middle: wp.array2d(dtype=wp.float64), side: wp.int32, cell: wp.int32, at: wp.vec3d,
           lead: wp.int32) -> wp.vec3d:
    """The way from the volume `cell` to the body at steps `at` whose fullest volume is `lead`, in calm lengths."""
    way = wp.vec3d()
    for axis in range(3):
        own = wp.float64(cell / stride_of(side, axis) % side)
        out = in_row(middle, side, axis, cell, at[axis]) - middle[axis, cell]
        back = in_row(middle, side, axis, lead, own) - in_row(middle, side, axis, lead, at[axis])
        way[axis] = (out - back) / wp.float64(2.0)
    return way


def middles(store: Store) -> wp.array:
    """Every place's middle along its row in each direction, (3, places), as the cells' sizes are now."""
    middle = spare(store, "middle", (3, store.side ** 3), wp.float64)
    wp.launch(_middles, dim=(3, store.side, store.side), device=middle.device, inputs=[
        store.cell_quanta, store.tile_home, store.tile_slots, store.side,
        wp.float64(QUANTA_A_BOX * store.across ** 3), middle])
    return middle
