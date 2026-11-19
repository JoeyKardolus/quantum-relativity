"""Fall: one step of the pull adding to the flow of the fluid it acts on, toward the deeper side.

The law, two lines. Along each direction a volume's fluid gains flow by its quanta times the
slope of the pull across it, times the step: the pull is read as a speed squared (as the
collapse limit reads it), so its slope is a gain of speed a turn and no number is set.

The slope is of the others' pull, never the volume's own handed back: at a side with a joint
it reads the neighbour's pull less what that very joint handed the neighbour (a sixth of the
joint's), at the neighbour's distance (`apart_of`); at a free side it reads what those
neighbours hand the volume itself (a twelfth of each reading), at no distance. The flow is
a whole number; the pull is read as it stands, so it acts a cell a step and no faster.
"""
import warp as wp

from physics.engine.axiom_3_structure.definitions.kinds import QUANTA_A_BOX
from physics.engine.axiom_3_structure.definitions.places import beside, joint_axis
from physics.engine.axiom_3_structure.definitions.store import Store, slot_at
from physics.engine.axiom_5_packing.definitions.pull_push import apart_of
from physics.engine.axiom_6_limits.definitions.collapse_limit import TOP_SPEED

wp.set_module_options({"enable_backward": False})  # nothing here is ever run backward


@wp.func
def _read(cell: wp.int32, side: wp.int32, axis: wp.int32, way: wp.int32, calm: wp.float64,
          tile_home: wp.array(dtype=wp.int32), tile_slots: wp.array(dtype=wp.int32),
          cell_quanta: wp.array(dtype=wp.int64), cell_pull: wp.array(dtype=wp.float32)) -> wp.vec2d:
    """The others' pull at the neighbour volume `way` along `axis`, and how far it stands; (0, 0): no joint."""
    joint = beside(cell, side, axis, way)
    if joint < 0 or cell_quanta[joint] == wp.int64(0) or slot_at(tile_home, tile_slots, joint) < 0:
        return wp.vec2d(wp.float64(0.0), wp.float64(0.0))
    there = beside(cell, side, axis, 2 * way)
    return wp.vec2d(wp.float64(cell_pull[there]) - wp.float64(cell_pull[joint]) / wp.float64(6.0),
                    apart_of(wp.float64(cell_quanta[cell]) / calm, wp.float64(cell_quanta[there]) / calm,
                             wp.abs(wp.float64(cell_quanta[joint])) / calm))


@wp.kernel
def _fall(own_cell: wp.array(dtype=wp.int32), tile_home: wp.array(dtype=wp.int32),
          tile_slots: wp.array(dtype=wp.int32), side: wp.int32, calm: wp.float64,
          a_turn: wp.float64, top: wp.float64, cell_quanta: wp.array(dtype=wp.int64),
          cell_pull: wp.array(dtype=wp.float32), cell_flow: wp.array2d(dtype=wp.int64)):
    cell = own_cell[wp.tid()]
    if cell_quanta[cell] <= wp.int64(0) or joint_axis(cell, side) >= 0:
        return  # the pull acts on a volume's fluid
    handed = wp.float64(0.0)  # what the neighbours hand this volume of the others' pull
    for axis in range(3):
        for way in range(-1, 2, 2):
            handed += _read(cell, side, axis, way, calm, tile_home, tile_slots, cell_quanta, cell_pull)[0] / wp.float64(12.0)
    for axis in range(3):
        low = _read(cell, side, axis, -1, calm, tile_home, tile_slots, cell_quanta, cell_pull)
        high = _read(cell, side, axis, 1, calm, tile_home, tile_slots, cell_quanta, cell_pull)
        if low[1] + high[1] > wp.float64(0.0):
            if low[1] == wp.float64(0.0):
                low[0] = handed
            if high[1] == wp.float64(0.0):
                high[0] = handed
            gain = wp.float64(cell_quanta[cell]) * (high[0] - low[0]) / (low[1] + high[1]) * a_turn
            most = wp.int64(wp.float64(cell_quanta[cell]) * top)  # nothing goes faster than the top speed
            cell_flow[cell, axis] = wp.clamp(cell_flow[cell, axis] + wp.int64(wp.floor(gain + wp.float64(0.5))), -most, most)


def fall(store: Store) -> None:
    """Add a step of the pull's slope to every own volume's flow."""
    if store.side == 0:
        return
    wp.launch(_fall, dim=store.count, device=store.box_quanta.device,
              inputs=[store.own_cell, store.tile_home, store.tile_slots, store.side,
                      wp.float64(QUANTA_A_BOX * store.across ** 3),
                      wp.float64(store.turn_per_step / (2.0 * wp.pi)), wp.float64(TOP_SPEED),
                      store.cell_quanta, store.cell_pull, store.cell_flow])
