"""Leak: one step of every volume's whirl leaking to every side. A property of the whirl: it always happens.

Each box of a volume leaks its share of a step (`leak_share`, counted a turn of a swirl) in
whole quanta toward one of the 26 directions, the one its own phase points to, whatever its
neighbours hold. At a free place the quanta gather, as that place's total, until they tear
off (in a bench, a store whose `leaks` is off, nothing gathers and that part stays). Where a volume stands they pass through the joint between (toward an edge or a corner:
by one of the sides that way, `leaned`) and arrive spread once round a core of that
volume's whirl, so it keeps its turn; with no joint between, nothing passes. What leaves one cell arrives in the other, to the
quantum, and takes its share of the volume's flow along. Shares and boxes are read off copies made before the launch, so a run repeats.
"""
import warp as wp

from physics.engine.axiom_1_energy.definitions.whole_move import odd_part
from physics.engine.axiom_3_structure.definitions.kinds import QUANTA_A_BOX
from physics.engine.axiom_3_structure.definitions.places import beside, joint_axis
from physics.engine.axiom_3_structure.definitions.store import Store, slot_at
from physics.engine.axiom_4_crossing.definitions.leak_share import leak_share, leaned, pointed, toward
from physics.engine.axiom_6_limits.definitions.collapse_limit import DEEP


@wp.kernel
def _leak(own_cell: wp.array(dtype=wp.int32), tile_home: wp.array(dtype=wp.int32),
          tile_slots: wp.array(dtype=wp.int32), held: wp.array(dtype=wp.int64),
          cell_pull: wp.array(dtype=wp.float32), deep: wp.float32,
          had: wp.array2d(dtype=wp.int32), side: wp.int32, gathers: wp.int32, calm: wp.float64,
          a_turn: wp.float64, cell_quanta: wp.array(dtype=wp.int64),
          flowed: wp.array2d(dtype=wp.int64), cell_flow: wp.array2d(dtype=wp.int64),
          box_quanta: wp.array2d(dtype=wp.int32), box_phase: wp.array2d(dtype=wp.float32)):
    slot, box = wp.tid()
    cell = own_cell[slot]
    if held[cell] <= wp.int64(0) or joint_axis(cell, side) >= 0 or cell_pull[cell] >= deep:
        return  # only a volume leaks, and past the limit nothing leaves it
    points = pointed(odd_part(box_phase[slot, box], 1))
    lean = leaned(wp.int32(points[0]), points[1])
    place = toward(cell, side, wp.int32(points[0]))
    if place >= 0 and slot_at(tile_home, tile_slots, place) >= 0:
        place = beside(cell, side, lean[0], 2 * lean[1])  # a volume stands there: by a side
    if place < 0:
        return  # past the room's end
    share = leak_share(wp.float64(held[cell]), calm) * a_turn
    moved = wp.min(had[slot, box], wp.int32(wp.floor(wp.float64(had[slot, box]) * share + points[1])))
    there = slot_at(tile_home, tile_slots, place)
    if there >= 0:  # a volume stands there: through the joint between
        if held[place] <= wp.int64(0) or held[(cell + place) / 2] >= wp.int64(0):
            return  # no joint between: nothing passes
        across = wp.int32(wp.cbrt(float(had.shape[1])) + 0.5)
        wp.atomic_add(box_quanta, there, box / (across * across) * across * across
                      + box % across * across + box / across % across, moved)  # once round its core
    elif gathers == 0:
        return  # a bench: nothing gathers outside it
    wp.atomic_sub(box_quanta, slot, box, moved)
    wp.atomic_add(cell_quanta, cell, wp.int64(-moved))
    wp.atomic_add(cell_quanta, place, wp.int64(moved))
    for along in range(3):  # what leaks takes its share of the volume's flow along
        carried = wp.int64(wp.float64(flowed[cell, along]) * wp.float64(moved) / wp.float64(held[cell]))
        wp.atomic_sub(cell_flow, cell, along, carried)
        wp.atomic_add(cell_flow, place, along, carried)


def leak(store: Store) -> None:
    """One step of the leak of every own volume; only in a universe that grows (`leaks`) does it gather at free places."""
    if store.side == 0:
        return  # a store that keeps no places has no sides
    wp.launch(_leak, dim=(store.count, store.across ** 3), device=store.box_quanta.device,
              inputs=[store.own_cell, store.tile_home, store.tile_slots,
                      wp.clone(store.cell_quanta), store.cell_pull, DEEP,
                      wp.clone(store.box_quanta), store.side, int(store.leaks),
                      wp.float64(QUANTA_A_BOX * store.across ** 3),
                      wp.float64(store.turn_per_step / (2.0 * wp.pi)),
                      store.cell_quanta, wp.clone(store.cell_flow), store.cell_flow,
                      store.box_quanta, store.box_phase])
