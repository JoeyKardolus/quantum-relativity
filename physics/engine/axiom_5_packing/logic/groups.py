"""Groups: every volume's pulling energy and every group's, written into the past kept, on the card.

One launch over the volumes' places: a volume's energy is its own plus and half of each
collapsed joint's at its sides (a place no volume stands at holds 0). Then one launch a
level: a group's energy is the sum of the eight below it, it stands at the middle of their
energy, and its lead is the fullest volume in it. Each body's energy goes into this step's
row of `cell_past`, so a group has a past of its own (`definitions/groups.py`).
"""
import warp as wp

from physics.engine.axiom_3_structure.definitions.kinds import QUANTA_A_BOX
from physics.engine.axiom_3_structure.definitions.places import stride_of
from physics.engine.axiom_3_structure.definitions.store import Store
from physics.engine.axiom_3_structure.logic.spare import spare
from physics.engine.axiom_5_packing.definitions.groups import (
    across_of, lead_node, node_of, nodes_of, wide_of)
from physics.engine.axiom_5_packing.logic.ways import plus_at

wp.set_module_options({"fuse_fp": False, "enable_backward": False})


@wp.kernel
def _volumes(cell_quanta: wp.array(dtype=wp.int64), tile_home: wp.array(dtype=wp.int32),
             tile_slots: wp.array(dtype=wp.int32), side: wp.int32, calm: wp.float64, row: wp.int32,
             energy: wp.array(dtype=wp.float64), stands: wp.array(dtype=wp.vec3d),
             lead: wp.array(dtype=wp.int32), past: wp.array2d(dtype=wp.float32)):
    first, second, third = wp.tid()
    at = wp.vec3i(first, second, third)
    node, cell = (at[0] * ((side + 1) / 2) + at[1]) * ((side + 1) / 2) + at[2], wp.int32(0)
    for axis in range(3):
        cell += 2 * at[axis] * stride_of(side, axis)
    own, sides = plus_at(cell, cell_quanta, tile_home, tile_slots, calm), wp.float64(0.0)
    for axis in range(3):
        for way in range(2):
            if own > wp.float64(0.0) and at[axis] + way > 0 and at[axis] + way < (side + 1) / 2:
                sides += plus_at(cell + (2 * way - 1) * stride_of(side, axis), cell_quanta, tile_home, tile_slots, calm)
    energy[node] = own + sides / wp.float64(2.0)
    stands[node] = wp.vec3d(wp.float64(2 * at[0]), wp.float64(2 * at[1]), wp.float64(2 * at[2]))
    lead[node] = cell
    past[row, node] = wp.float32(energy[node])


@wp.kernel
def _groups(wide: wp.int32, level: wp.int32, row: wp.int32, energy: wp.array(dtype=wp.float64),
            stands: wp.array(dtype=wp.vec3d), lead: wp.array(dtype=wp.int32),
            past: wp.array2d(dtype=wp.float32)):
    first, second, third = wp.tid()
    at = wp.vec3i(first, second, third)
    node, below = node_of(wide, level, at), across_of(wide, level - 1)
    held, where, most, fullest = wp.float64(0.0), wp.vec3d(), wp.float64(0.0), wp.int32(0)
    for part in range(8):
        under = wp.vec3i(2 * at[0] + part / 4, 2 * at[1] + part / 2 % 2, 2 * at[2] + part % 2)
        if wp.max(under) < below:
            one = node_of(wide, level - 1, under)
            held += energy[one]
            where += energy[one] * stands[one]
            if energy[lead_node(wide, lead[one])] > most:
                most = wp.float64(energy[lead_node(wide, lead[one])])
                fullest = wp.int32(lead[one])
    energy[node] = held
    lead[node] = fullest
    stands[node] = where / wp.max(held, wp.float64(1.0e-300))
    past[row, node] = wp.float32(held)


def bodies(store: Store, top: int, row: int) -> tuple:
    """Write every body's energy (also into row `row` of the past), where it stands and its lead; gives the three."""
    wide, nodes, device = wide_of(store.side), nodes_of(store.side, top), store.box_quanta.device
    energy, stands, lead = (spare(store, name, nodes, kind) for name, kind in
                            (("energy", wp.float64), ("stands", wp.vec3d), ("lead", wp.int32)))
    wp.launch(_volumes, dim=(wide, wide, wide), device=device, inputs=[
        store.cell_quanta, store.tile_home, store.tile_slots, store.side,
        wp.float64(QUANTA_A_BOX * store.across ** 3), row, energy, stands, lead, store.cell_past])
    for level in range(1, top + 1):
        wp.launch(_groups, dim=((wide + (1 << level) - 1) >> level,) * 3, device=device,
                  inputs=[wide, level, row, energy, stands, lead, store.cell_past])
    return energy, stands, lead
