"""Cross: one step of plus energy crossing every joint that lies between two volumes.

The cells stand in a fixed line, volumes at the even places and joints at the odd, so a
joint's two volumes follow from its place. A volume is laid as the mirror of the volume
before it (the other sense, like meshing gears), so two calm volumes' facing fluid
agrees. Each box of the left volume that faces the joint moves with the box of the right
volume facing it, by the pair rotation at the joint's `passing_turn`: energy goes the
way its phase carries it and arrives with its phase, so it is still moving. Friction
then takes its share of the motion across (the step in phase between the two boxes), and
that share of the whole quanta that crossed is booked as the taker's heat. What one
volume loses the other gains; the joint's quanta stay.
"""
import warp as wp

from physics.engine.axiom_1_energy.definitions.draw_uniform import draw_uniform
from physics.engine.axiom_1_energy.definitions.pair_rotation import pair_rotation, phase_of, wave_of
from physics.engine.axiom_1_energy.definitions.whole_move import whole_move
from physics.engine.axiom_3_structure.definitions.store import slot_at
from physics.engine.axiom_4_crossing.definitions.passage import (
    less_step, passing_turn, row_minus, rub_share)

wp.set_module_options({"fuse_fp": False})


@wp.kernel
def _cross(own_cell: wp.array(dtype=wp.int32), tile_home: wp.array(dtype=wp.int32),
           tile_slots: wp.array(dtype=wp.int32), across: wp.int32, seed: wp.uint32,
           step: wp.uint32, cell_quanta: wp.array(dtype=wp.int32),
           cell_heat: wp.array(dtype=wp.int32),
           box_quanta: wp.array2d(dtype=wp.int32), box_phase: wp.array2d(dtype=wp.float32)):
    slot, first, third = wp.tid()
    row = first * across * across + third  # the row's first box: where the line's direction is 0
    joint = own_cell[slot]
    if joint % 2 == 0 or joint + 1 >= cell_quanta.shape[0]:
        return  # only a cell at an odd place has a volume on both sides
    left = slot_at(tile_home, tile_slots, joint - 1)
    right = slot_at(tile_home, tile_slots, joint + 1)
    minus = row_minus(box_quanta, slot, row, across)
    if left < 0 or right < 0 or minus <= wp.float64(0.0):
        return  # a calm volume is not crossed into yet; with no minus between, nothing passes
    low = row + (across - 1) * across  # the left volume's box that faces the joint
    low_has = box_quanta[left, low]
    high_has = box_quanta[right, row]
    rub = rub_share(minus, low_has, high_has)
    low_wave, high_wave = pair_rotation(wave_of(low_has, box_phase[left, low]),
                                        wave_of(high_has, box_phase[right, row]),
                                        passing_turn(rub))
    box = wp.uint64(joint) * wp.uint64(box_quanta.shape[1]) + wp.uint64(row)
    flow = wp.dot(high_wave, high_wave) - wp.float64(high_has)  # what the right volume gains
    moved = whole_move(flow, draw_uniform(seed, step, wp.uint32(6), box), low_has, high_has)
    rubbed = wp.int32(wp.floor(rub * wp.float64(wp.abs(moved))
                               + draw_uniform(seed, step, wp.uint32(7), box)))
    step_across = less_step(phase_of(low_wave, box_phase[left, low]),
                            phase_of(high_wave, box_phase[right, row]), rub)
    box_quanta[left, low] = low_has - moved
    box_quanta[right, row] = high_has + moved
    box_phase[left, low] = step_across[0]
    box_phase[right, row] = step_across[1]
    if moved > 0:
        wp.atomic_add(cell_heat, joint + 1, rubbed)
    if moved < 0:
        wp.atomic_add(cell_heat, joint - 1, rubbed)
    wp.atomic_add(cell_quanta, joint - 1, -moved)
    wp.atomic_add(cell_quanta, joint + 1, moved)


def cross(store, before=None) -> None:
    """One step of crossing at every own joint; `before()` lets a watcher look first."""
    if before:
        before()
    wp.launch(_cross, dim=(store.count, store.across, store.across), device=store.box_quanta.device,
              inputs=[store.own_cell, store.tile_home, store.tile_slots, store.across, store.seed,
                      store.step, store.cell_quanta, store.cell_heat, store.box_quanta,
                      store.box_phase])
