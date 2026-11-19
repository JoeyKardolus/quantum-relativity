"""Tear: where enough leaked energy has gathered at a free place, the leak has torn a volume off.

Leaked quanta gather at a free place as its total (`leak.py`); they are no cell yet and do
nothing. Once every part of them could be a whirl (`PARTS` times the least whirl) the
place is a new energy volume, its gathering shared in seven: its own part and one toward
each side. Toward a side where a volume stands, or forms in the same step, that part flips
to minus and is the joint between (two that form side by side both give). Plus lost is
minus gained, to the quantum (`torn.py`: the law place by place, for checks). On the card:
a launch over the places finds the ripe ones and one count comes back; only when there are
any, a second shares them out and `seat` gives the new cells their boxes, as whole whirls.
"""
import warp as wp

from physics.engine.axiom_3_structure.definitions.places import joint_axis, joint_ends, stride_of
from physics.engine.axiom_3_structure.definitions.store import Store, slot_at
from physics.engine.axiom_3_structure.logic.seat import seat
from physics.engine.axiom_3_structure.logic.spare import spare
from physics.engine.axiom_6_limits.definitions.least_whirl import PARTS, least_whirl


@wp.kernel
def _ripe(cell_quanta: wp.array(dtype=wp.int64), tile_home: wp.array(dtype=wp.int32),
          tile_slots: wp.array(dtype=wp.int32), enough: wp.float64, side: wp.int32,
          ripe: wp.array(dtype=wp.int64), seen: wp.array(dtype=wp.int32)):
    cell = wp.tid()
    ripe[cell] = wp.int64(0)
    if wp.float64(cell_quanta[cell]) < enough or slot_at(tile_home, tile_slots, cell) >= 0:
        return
    ripe[cell] = cell_quanta[cell]
    wp.atomic_add(seen, 0, 1)
    for axis in range(3):
        at = cell / stride_of(side, axis) % side
        if at <= 1 or at >= side - 2:
            wp.atomic_add(seen, 1, 1)  # on the room's last layer


@wp.func
def _stands(cell: wp.int32, ripe: wp.array(dtype=wp.int64), tile_home: wp.array(dtype=wp.int32),
            tile_slots: wp.array(dtype=wp.int32)) -> wp.bool:  # a cell stands there, or forms in this step
    return ripe[cell] > wp.int64(0) or slot_at(tile_home, tile_slots, cell) >= 0


@wp.kernel
def _torn(ripe: wp.array(dtype=wp.int64), tile_home: wp.array(dtype=wp.int32),
          tile_slots: wp.array(dtype=wp.int32), side: wp.int32, parts: wp.int64,
          cell_quanta: wp.array(dtype=wp.int64), fresh: wp.array(dtype=wp.int32)):
    cell, made, sides, axis = wp.tid(), wp.int64(0), wp.int64(0), joint_axis(wp.tid(), side)
    if axis >= 0:  # a joint's place: a part of each ripe end whose other end stands
        ends = joint_ends(cell, side, axis)
        for end in range(2):
            if ripe[ends[end]] > wp.int64(0) and _stands(ends[1 - end], ripe, tile_home, tile_slots):
                made -= ripe[ends[end]] / parts
    elif ripe[cell] > wp.int64(0):
        for along in range(3):
            for way in range(2):
                if _stands(cell + (4 * way - 2) * stride_of(side, along), ripe, tile_home, tile_slots):
                    sides += wp.int64(1)
        made = ripe[cell] - sides * (ripe[cell] / parts)
    fresh[cell] = wp.int32(made != wp.int64(0))
    if made != wp.int64(0):
        cell_quanta[cell] = made


def tear(store: Store) -> None:
    """Make every cell the leak tears off in this step, each with its own boxes."""
    if not store.leaks:
        return
    places, device, enough = store.side ** 3, store.cell_quanta.device, wp.float64(PARTS * least_whirl(store.across))
    ripe, seen, fresh = (spare(store, name, size, kind) for name, size, kind in (
        ("ripe", places, wp.int64), ("ripe_seen", 2, wp.int32), ("fresh", places, wp.int32)))
    seen.zero_()
    wp.launch(_ripe, dim=places, device=device, inputs=[
        store.cell_quanta, store.tile_home, store.tile_slots, enough, store.side, ripe, seen])
    found, at_end = seen.numpy()
    if at_end:
        raise RuntimeError(f"no room for another layer: the store holds {store.side // 4}")
    if found:
        wp.launch(_torn, dim=places, device=device, inputs=[
            ripe, store.tile_home, store.tile_slots, store.side, wp.int64(PARTS), store.cell_quanta, fresh])
        seat(store, fresh)
