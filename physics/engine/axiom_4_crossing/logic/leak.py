"""Leak: one step of every volume's whirl leaking toward the places round it.

Each box of a volume leaks its share of a step (`leak_share`, counted a turn of a swirl) in
whole quanta toward one of the 26 directions. Toward a volume nothing is done here: there
the leak is the crossing through the joint between them (`cross.py`), and where no joint
lies between two volumes nothing passes. Toward a free place the quanta leave the volume
and gather there, as that place's total: plus that leaves a volume arrives at the place,
to the quantum. The shares are read off the totals as they were before the launch.
"""
import warp as wp

from physics.engine.axiom_1_energy.definitions.draw_uniform import draw_uniform
from physics.engine.axiom_3_structure.definitions.kinds import QUANTA_A_BOX
from physics.engine.axiom_3_structure.definitions.store import Store, slot_at
from physics.engine.axiom_4_crossing.definitions.leak_share import leak_share, toward


@wp.kernel
def _leak(own_cell: wp.array(dtype=wp.int32), tile_home: wp.array(dtype=wp.int32),
          tile_slots: wp.array(dtype=wp.int32), held: wp.array(dtype=wp.int32),
          side: wp.int32, calm: wp.float64, a_turn: wp.float64, seed: wp.uint32,
          step: wp.uint32, cell_quanta: wp.array(dtype=wp.int32),
          box_quanta: wp.array2d(dtype=wp.int32)):
    slot, box = wp.tid()
    cell = own_cell[slot]
    if held[cell] <= 0:
        return  # only a volume leaks
    key = wp.uint64(cell) * wp.uint64(box_quanta.shape[1]) + wp.uint64(box)
    place = toward(cell, side, draw_uniform(seed, step, wp.uint32(9), key))
    if place < 0 or slot_at(tile_home, tile_slots, place) >= 0:
        return  # a volume stands there: the joint between carries the leak
    share = leak_share(wp.float64(held[cell]), calm) * a_turn
    has = box_quanta[slot, box]
    moved = wp.min(has, wp.int32(wp.floor(
        wp.float64(has) * share + draw_uniform(seed, step, wp.uint32(10), key))))
    box_quanta[slot, box] = has - moved
    wp.atomic_add(cell_quanta, cell, -moved)
    wp.atomic_add(cell_quanta, place, moved)


def leak(store: Store) -> None:
    """One step of the leak of every own volume, where the store's volumes leak."""
    if not store.leaks:
        return
    wp.launch(_leak, dim=(store.count, store.across ** 3), device=store.box_quanta.device,
              inputs=[store.own_cell, store.tile_home, store.tile_slots,
                      wp.clone(store.cell_quanta), store.side,
                      wp.float64(QUANTA_A_BOX * store.across ** 3),
                      wp.float64(store.turn_per_step / (2.0 * wp.pi)), store.seed, store.step,
                      store.cell_quanta, store.box_quanta])
