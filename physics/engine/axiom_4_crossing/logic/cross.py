"""Cross: plus energy goes from volume to volume through the joint between them, by the fluid's own step.

Contract: `cross(state, settings)` copies `cell_quanta` into `quanta_before`, counts how
many acting ways share each mouth, then for pass 0 to `way_passes - 1` takes every
living way of that pass (so no mouth is in two ways at once) in four launches. A way acts
only when its two ends are energy volumes and its joint is a joint that holds minus
energy: energy crosses only through minus. The two mouths of a way face each other, box
for box (the n-th box of one and the n-th of the other, `mouth_box`).
  trade  each facing pair of boxes takes one step as a pair (`pair_rotation`, axiom 1):
         exactly the step two boxes inside a cell take, on the way's local time
         (`crossing_clock`), made whole by `whole_move`. Fluid flows from the box whose
         phase is ahead, a fuller box evens out with an emptier one, and two equal
         waves are left exactly as they are. A mouth that several ways share (after a
         tear) splits that local time evenly between them.
  rub    `friction_law`: a share of the quanta that cross lose their direction in the joint.
  move   the two boxes give and take the whole quanta; the giving box gets the step's
         phase, the taking box its phase by `arrival_law`: what lost its direction
         arrives at rest in the fluid it joins.
  book   the way's ledgers, the joint's `crossed` and `heated`, and the heat that left
         one volume and arrived in the other (`heat_leaving`, `heat_made`).
Writes `quanta` and `phase` of mouth boxes, `way_out_a`, `way_out_b`, `way_lost_a`,
`way_lost_b`, `way_moved`, `crossed` and `heated` of the joint, `heat_quanta` of the two
volumes, `quanta_before`, `heat_before`. Nothing else: no box of a joint, no box inside
a volume, no total. The step makes nothing and drives nothing: the energy is one fluid,
inside a cell and from cell to cell, and a universe of one phase and even energy stays
exactly as it is. No box ever changes cell: only quanta do.
Reads `settings.step_time`, `settings.friction`.
"""
from __future__ import annotations

import warp as wp

from physics.engine.axiom_1_energy.definitions.cell_sum import cell_sum
from physics.engine.axiom_1_energy.definitions.draw_uniform import draw_uniform
from physics.engine.axiom_1_energy.definitions.pair_rotation import pair_rotation
from physics.engine.axiom_1_energy.definitions.passes import WAY_FRICTION, WAY_ROUND
from physics.engine.axiom_1_energy.definitions.whole_move import whole_move
from physics.engine.axiom_3_structure.definitions.kinds import ENERGY_VOLUME, JOINT
from physics.engine.axiom_4_crossing.definitions.arrival_law import arrival_law
from physics.engine.axiom_4_crossing.definitions.crossing_clock import crossing_clock
from physics.engine.axiom_4_crossing.definitions.friction_law import friction_law, heat_leaving, heat_made
from physics.engine.axiom_4_crossing.definitions.mouth import mouth_box, mouth_size
from physics.engine.axiom_7_quantum.definitions.wrap_phase import wrap_phase

wp.set_module_options({"fuse_fp": False})

_ENERGY_VOLUME = wp.constant(wp.int32(ENERGY_VOLUME))
_JOINT = wp.constant(wp.int32(JOINT))
_WAY_ROUND = wp.constant(wp.uint32(WAY_ROUND))
_WAY_FRICTION = wp.constant(wp.uint32(WAY_FRICTION))
_PLACES = 16
_SIDES = 2
_DIRECTIONS = 27


@wp.func
def _way_lives(slot: wp.int32, way_joint: wp.array(dtype=wp.int32), way_a: wp.array(dtype=wp.int32),
               way_b: wp.array(dtype=wp.int32), kind: wp.array(dtype=wp.int32),
               cell_quanta: wp.array(dtype=wp.int64)) -> bool:
    """Whether this slot holds a way between two energy volumes through a joint that holds minus energy."""
    joint = way_joint[slot]
    if joint < 0:
        return False
    if kind[way_a[slot]] != _ENERGY_VOLUME or kind[way_b[slot]] != _ENERGY_VOLUME or kind[joint] != _JOINT:
        return False
    return cell_quanta[joint] > wp.int64(0)


@wp.kernel
def _count_mouths(way_a: wp.array(dtype=wp.int32), way_b: wp.array(dtype=wp.int32),
                  way_mouth_a: wp.array(dtype=wp.int32), way_mouth_b: wp.array(dtype=wp.int32),
                  way_joint: wp.array(dtype=wp.int32), kind: wp.array(dtype=wp.int32),
                  cell_quanta: wp.array(dtype=wp.int64), mouth_ways: wp.array2d(dtype=wp.int32)):
    """One thread per way slot: count the acting way on each of its two mouths."""
    slot = wp.tid()
    if not _way_lives(slot, way_joint, way_a, way_b, kind, cell_quanta):
        return
    wp.atomic_add(mouth_ways, way_a[slot], way_mouth_a[slot], 1)
    wp.atomic_add(mouth_ways, way_b[slot], way_mouth_b[slot], 1)


@wp.kernel
def _hold_now(quanta: wp.array4d(dtype=wp.int32), heat_quanta: wp.array(dtype=wp.int64),
              pass_quanta: wp.array(dtype=wp.int64), heat_before: wp.array(dtype=wp.int64)):
    """One thread per cell: what it holds, and how much of that is heat, as this pass begins."""
    cell = wp.tid()
    pass_quanta[cell] = cell_sum(quanta, cell)
    heat_before[cell] = heat_quanta[cell]


@wp.func
def _acts(slot: wp.int32, place: wp.int32, this_pass: wp.int32, way_pass: wp.array(dtype=wp.int32),
          way_mouth_a: wp.array(dtype=wp.int32), way_joint: wp.array(dtype=wp.int32),
          way_a: wp.array(dtype=wp.int32), way_b: wp.array(dtype=wp.int32), kind: wp.array(dtype=wp.int32),
          cell_quanta: wp.array(dtype=wp.int64)) -> bool:
    """Whether this place is a facing pair of boxes of a living way of this pass."""
    if way_pass[slot] != this_pass or not _way_lives(slot, way_joint, way_a, way_b, kind, cell_quanta):
        return False
    return place < mouth_size(way_mouth_a[slot])


@wp.func
def _pair_index(slot: wp.int32, place: wp.int32) -> wp.uint64:
    """The draw index of one facing pair of boxes: its way's slot times 16 plus its place."""
    return wp.uint64(slot) * wp.uint64(_PLACES) + wp.uint64(place)


@wp.kernel
def _trade(this_pass: wp.int32, seed: wp.uint32, step: wp.uint32, step_time: wp.float64,
           way_a: wp.array(dtype=wp.int32), way_b: wp.array(dtype=wp.int32),
           way_mouth_a: wp.array(dtype=wp.int32), way_mouth_b: wp.array(dtype=wp.int32),
           way_joint: wp.array(dtype=wp.int32), way_pass: wp.array(dtype=wp.int32),
           kind: wp.array(dtype=wp.int32), cell_quanta: wp.array(dtype=wp.int64),
           rate: wp.array(dtype=wp.float32), mouth_ways: wp.array2d(dtype=wp.int32),
           quanta: wp.array4d(dtype=wp.int32), phase: wp.array4d(dtype=wp.float64),
           moved: wp.array2d(dtype=wp.int64), stepped: wp.array3d(dtype=wp.float64)):
    """One thread per facing pair of boxes: their step as a pair, as whole quanta a to b and two new phases."""
    slot, place = wp.tid()
    if not _acts(slot, place, this_pass, way_pass, way_mouth_a, way_joint, way_a, way_b, kind, cell_quanta):
        return
    cell_a = way_a[slot]
    cell_b = way_b[slot]
    box_a = mouth_box(way_mouth_a[slot], place)
    box_b = mouth_box(way_mouth_b[slot], place)
    sharing = wp.max(mouth_ways[cell_a, way_mouth_a[slot]], mouth_ways[cell_b, way_mouth_b[slot]])
    turn = crossing_clock(step_time, wp.float64(rate[cell_a]), wp.float64(rate[cell_b]),
                          wp.float64(rate[way_joint[slot]])) / wp.float64(sharing)
    has_a = wp.int64(quanta[cell_a, box_a[0], box_a[1], box_a[2]])
    has_b = wp.int64(quanta[cell_b, box_b[0], box_b[1], box_b[2]])
    left_a, phase_a, left_b, phase_b = pair_rotation(
        wp.float64(has_a), phase[cell_a, box_a[0], box_a[1], box_a[2]],
        wp.float64(has_b), phase[cell_b, box_b[0], box_b[1], box_b[2]], turn)
    moved[slot, place] = whole_move(left_b - wp.float64(has_b),
                                    draw_uniform(seed, step, _WAY_ROUND, _pair_index(slot, place)), has_a, has_b)
    stepped[slot, place, 0] = phase_a
    stepped[slot, place, 1] = phase_b


@wp.kernel
def _rub(seed: wp.uint32, step: wp.uint32, friction: wp.float64, moved: wp.array2d(dtype=wp.int64),
         lost: wp.array2d(dtype=wp.int64)):
    """One thread per facing pair of boxes: how many of the quanta that cross lose their direction in the joint."""
    slot, place = wp.tid()
    lost[slot, place] = friction_law(wp.abs(moved[slot, place]), friction,
                                     draw_uniform(seed, step, _WAY_FRICTION, _pair_index(slot, place)))


@wp.func
def _give_box(quanta: wp.array4d(dtype=wp.int32), phase: wp.array4d(dtype=wp.float64), cell: wp.int32,
              box: wp.vec3i, giving: wp.int64, stepped: wp.float64):
    """One box gives `giving` quanta and gets the phase the pair's step left it."""
    quanta[cell, box[0], box[1], box[2]] -= wp.int32(giving)
    phase[cell, box[0], box[1], box[2]] = wrap_phase(stepped)


@wp.func
def _take_box(quanta: wp.array4d(dtype=wp.int32), phase: wp.array4d(dtype=wp.float64), cell: wp.int32,
              box: wp.vec3i, taking: wp.int64, lost: wp.int64, stepped: wp.float64):
    """One box takes `taking` quanta, `lost` of them without their direction."""
    has = quanta[cell, box[0], box[1], box[2]]
    standing = phase[cell, box[0], box[1], box[2]]
    if has == 0:
        standing = stepped
    phase[cell, box[0], box[1], box[2]] = arrival_law(wp.float64(has), standing, wrap_phase(stepped - standing),
                                                      wp.float64(taking), wp.float64(lost))
    quanta[cell, box[0], box[1], box[2]] = has + wp.int32(taking)


@wp.kernel
def _move(this_pass: wp.int32, way_a: wp.array(dtype=wp.int32), way_b: wp.array(dtype=wp.int32),
          way_mouth_a: wp.array(dtype=wp.int32), way_mouth_b: wp.array(dtype=wp.int32),
          way_joint: wp.array(dtype=wp.int32), way_pass: wp.array(dtype=wp.int32),
          kind: wp.array(dtype=wp.int32), cell_quanta: wp.array(dtype=wp.int64),
          moved: wp.array2d(dtype=wp.int64), stepped: wp.array3d(dtype=wp.float64),
          lost: wp.array2d(dtype=wp.int64), quanta: wp.array4d(dtype=wp.int32),
          phase: wp.array4d(dtype=wp.float64)):
    """One thread per facing pair of boxes: the two boxes give and take what their step moved."""
    slot, place = wp.tid()
    if not _acts(slot, place, this_pass, way_pass, way_mouth_a, way_joint, way_a, way_b, kind, cell_quanta):
        return
    box_a = mouth_box(way_mouth_a[slot], place)
    box_b = mouth_box(way_mouth_b[slot], place)
    crossing = moved[slot, place]
    if crossing >= wp.int64(0):
        _give_box(quanta, phase, way_a[slot], box_a, crossing, stepped[slot, place, 0])
        _take_box(quanta, phase, way_b[slot], box_b, crossing, lost[slot, place], stepped[slot, place, 1])
    else:
        _give_box(quanta, phase, way_b[slot], box_b, -crossing, stepped[slot, place, 1])
        _take_box(quanta, phase, way_a[slot], box_a, -crossing, lost[slot, place], stepped[slot, place, 0])


@wp.func
def _sent(moved: wp.array2d(dtype=wp.int64), counted: wp.array2d(dtype=wp.int64), slot: wp.int32,
          sense: wp.int64) -> wp.int64:
    """The sum of `counted` over the way's pairs whose quanta went in one sense (1: a to b, -1: b to a)."""
    total = wp.int64(0)
    for place in range(_PLACES):
        if moved[slot, place] * sense > wp.int64(0):
            total += wp.abs(counted[slot, place])
    return total


@wp.kernel
def _book(this_pass: wp.int32, way_a: wp.array(dtype=wp.int32), way_b: wp.array(dtype=wp.int32),
          way_joint: wp.array(dtype=wp.int32), way_pass: wp.array(dtype=wp.int32),
          kind: wp.array(dtype=wp.int32), cell_quanta: wp.array(dtype=wp.int64),
          moved: wp.array2d(dtype=wp.int64), lost: wp.array2d(dtype=wp.int64),
          pass_quanta: wp.array(dtype=wp.int64), heat_before: wp.array(dtype=wp.int64),
          way_out_a: wp.array(dtype=wp.int32), way_out_b: wp.array(dtype=wp.int32),
          way_lost_a: wp.array(dtype=wp.int32), way_lost_b: wp.array(dtype=wp.int32),
          way_moved: wp.array(dtype=wp.int32), crossed: wp.array(dtype=wp.int64),
          heated: wp.array(dtype=wp.int64), heat_quanta: wp.array(dtype=wp.int64)):
    """One thread per way slot: book what crossed each way, what lost its direction, and the heat that went with it."""
    slot = wp.tid()
    if way_pass[slot] != this_pass or not _way_lives(slot, way_joint, way_a, way_b, kind, cell_quanta):
        return
    out_a = _sent(moved, moved, slot, wp.int64(1))
    out_b = _sent(moved, moved, slot, wp.int64(-1))
    lost_a = _sent(moved, lost, slot, wp.int64(1))
    lost_b = _sent(moved, lost, slot, wp.int64(-1))
    heat_a = heat_leaving(out_a, heat_before[way_a[slot]], pass_quanta[way_a[slot]])
    heat_b = heat_leaving(out_b, heat_before[way_b[slot]], pass_quanta[way_b[slot]])
    way_out_a[slot] = wp.int32(out_a)
    way_out_b[slot] = wp.int32(out_b)
    way_lost_a[slot] = wp.int32(lost_a)
    way_lost_b[slot] = wp.int32(lost_b)
    way_moved[slot] = wp.int32(out_a - out_b)
    wp.atomic_add(heat_quanta, way_b[slot], heat_a + heat_made(out_a, heat_a, lost_a) - heat_b)
    wp.atomic_add(heat_quanta, way_a[slot], heat_b + heat_made(out_b, heat_b, lost_b) - heat_a)
    wp.atomic_add(crossed, way_joint[slot], out_a + out_b)
    wp.atomic_add(heated, way_joint[slot], lost_a + lost_b)


def cross(state, settings) -> None:
    """One step's crossing of every living way, pass by pass: trade, rub, move, book."""
    device, ways = state.device, int(state.way_count)
    wp.copy(state.quanta_before, state.cell_quanta)
    if ways == 0:
        return
    seed, step = wp.uint32(state.seed), wp.uint32(state.step)
    pass_quanta = wp.zeros(state.capacity, dtype=wp.int64, device=device)
    mouth_ways = wp.zeros((state.capacity, _DIRECTIONS), dtype=wp.int32, device=device)
    stepped = wp.zeros((ways, _PLACES, _SIDES), dtype=wp.float64, device=device)
    lost = wp.zeros((ways, _PLACES), dtype=wp.int64, device=device)
    ends = [state.way_a, state.way_b, state.way_mouth_a, state.way_mouth_b, state.way_joint]
    living = [state.way_pass, state.kind, state.cell_quanta]
    wp.launch(_count_mouths, dim=ways, device=device, inputs=[*ends, state.kind, state.cell_quanta, mouth_ways])
    for this_pass in range(int(state.way_passes)):
        moved = wp.zeros((ways, _PLACES), dtype=wp.int64, device=device)
        wp.launch(_hold_now, dim=int(state.count), device=device,
                  inputs=[state.quanta, state.heat_quanta, pass_quanta, state.heat_before])
        wp.launch(_trade, dim=(ways, _PLACES), device=device,
                  inputs=[this_pass, seed, step, wp.float64(settings.step_time), *ends, *living, state.rate,
                          mouth_ways, state.quanta, state.phase, moved, stepped])
        wp.launch(_rub, dim=(ways, _PLACES), device=device,
                  inputs=[seed, step, wp.float64(settings.friction), moved, lost])
        wp.launch(_move, dim=(ways, _PLACES), device=device,
                  inputs=[this_pass, *ends, *living, moved, stepped, lost, state.quanta, state.phase])
        wp.launch(_book, dim=ways, device=device,
                  inputs=[this_pass, state.way_a, state.way_b, state.way_joint, *living, moved, lost, pass_quanta,
                          state.heat_before, state.way_out_a, state.way_out_b, state.way_lost_a, state.way_lost_b,
                          state.way_moved, state.crossed, state.heated, state.heat_quanta])
