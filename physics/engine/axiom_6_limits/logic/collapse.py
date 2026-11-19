"""Collapse: a joint whose two volumes are both past the limit flips to plus where it stands.

On the card, one launch over the own cells, after the tear and before the fluid moves,
from the pull of the step before: where nothing at the top speed could climb out on either
side of a joint, it has nothing left to hold apart. Its minus quanta flip to plus, the same
amount, laid as whole ring whirls turning a volume's way, and from then on it and its two
volumes are one volume: plus beside plus with no joint between. Nothing else changes. Every
cell keeps its place, its own whole number and its other joints, so no number grows at a
merge (the merged volume's energy is its cells' sum) and cells far away notice only the
deeper pull. How many fell and what they flipped comes back as two numbers.
"""
import warp as wp

from physics.engine.axiom_3_structure.definitions.places import joint_axis, joint_ends
from physics.engine.axiom_3_structure.definitions.store import Store
from physics.engine.axiom_3_structure.logic.spare import spare
from physics.engine.axiom_3_structure.logic.take_own import deal
from physics.engine.axiom_6_limits.definitions.collapse_limit import TOP_SPEED

wp.set_module_options({"fuse_fp": False})


@wp.kernel
def _collapse(own_cell: wp.array(dtype=wp.int32), side: wp.int32, speed_squared: wp.float32,
              cell_pull: wp.array(dtype=wp.float32), turned: wp.float64,
              pattern: wp.array2d(dtype=wp.vec2d), cell_quanta: wp.array(dtype=wp.int64),
              own_turned: wp.array(dtype=wp.float32), box_quanta: wp.array2d(dtype=wp.int32),
              box_phase: wp.array2d(dtype=wp.float32), fell: wp.array(dtype=wp.int64)):
    slot = wp.tid()
    joint = own_cell[slot]
    axis = joint_axis(joint, side)
    if axis < 0 or cell_quanta[joint] >= wp.int64(0):
        return  # only a live joint can fall
    ends = joint_ends(joint, side, axis)
    if wp.float32(2.0) * wp.min(cell_pull[ends[0]], cell_pull[ends[1]]) < speed_squared:
        return  # one of its volumes can still be climbed out of
    held = -cell_quanta[joint]
    cell_quanta[joint] = held
    deal(slot, held, turned, pattern, box_quanta, box_phase)
    own_turned[slot] = 0.0  # laid fresh: nothing turned past calm yet
    wp.atomic_add(fell, 0, wp.int64(1))
    wp.atomic_add(fell, 1, held)


def collapse(store: Store) -> None:
    """Every joint past the limit flips to plus where it stands; the books stay exact."""
    if store.side == 0 or store.count == 0:
        return
    fell = spare(store, "fell", 2, wp.int64)
    fell.zero_()
    wp.launch(_collapse, dim=store.count, device=store.box_quanta.device, inputs=[
        store.own_cell, store.side, TOP_SPEED ** 2, store.cell_pull,
        wp.float64(store.turn_per_step * store.step), store.pattern, store.cell_quanta,
        store.own_turned, store.box_quanta, store.box_phase, fell])
    joints, flipped = fell.numpy()
    store.collapses, store.flipped = store.collapses + int(joints), store.flipped + int(flipped)
