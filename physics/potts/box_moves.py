"""Box moves: cells grow and shrink by claiming or giving up Planck boxes at their borders.

Contract: `box_moves(universe, settings)` runs one sweep of eight move phases on the GPU
and returns the number of boxes that changed owner (one small copy to the host). A
phase visits the boxes whose coordinates have one parity pattern (x, y, z mod 2), so
two boxes moving at once are at least two boxes apart along some axis and neither
reads the other's 3x3x3 surroundings: neighbouring moves never conflict. A visited box
x owned by cell s (or empty, s = -1) may switch to a face neighbour's cell t != s:
  1. Cost. Of its face neighbours' cells it picks the t with the lowest change in
     H = sum over cells of stiffness * E * (n / want - 1)**2   (size follows energy)
       - for joints, the work of their push P = push * (want / n)**push_power
       + for energy volumes, E * (mean potential over its boxes)     (the pull)
     and switches only when that change is below 0. n, E, want and the mean potential
     are the cell's (cell_sums.py, refreshed each step; counts updated live). So an
     energy volume reaches for boxes where the pull is deeper (towards other A's,
     through the joints between) and a joint resists losing boxes, steeply as it is
     squeezed.
  2. Time. It then happens with probability min(clock of s, clock of t) / clock_max
     (bible: time is local; an empty box runs at clock_max = (s + 1) / s).
  3. Rules. Every other cell touching x across a face must already touch t (neighbours
     never change: no move makes a new contact or ends an existing one) and be of the
     other kind (ABAB: an A touches only B's); s must stay connected (its face
     neighbours of x are joined inside x's 3x3x3 surroundings without x) and keep at
     least one box; s may lose and t gain at most max(1, move_share * n) boxes per
     phase. The contact table's face counts are updated atomically; a move that would
     drop any count to 0 is undone whole.
  4. Energy. The box's energy and momentum go, exactly, to one of its face neighbours
     still in s (the first in face order); then the first face neighbour in t hands
     it a third of its own energy and the same share of its momentum (a third: at
     most two movers share one neighbour in a phase, so it never goes negative). A
     move changes only sizes, never a cell's energy; total energy stays exact; the
     cell stretches into the box rather than leaving it a vacuum.
A box never becomes empty: space only grows, as cells claim empty edge boxes.
Atomics make which of two racing moves wins, and so a run, not exactly repeatable.
"""
from __future__ import annotations

import warp as wp

from physics.potts.box_owner import box_owner
from physics.potts.direction import direction
from physics.potts.settings import Settings
from physics.potts.universe import QUANTUM, Universe

_QUANTUM = wp.constant(QUANTUM)
_FACE_BITS = wp.constant((1 << 4) | (1 << 22) | (1 << 10) | (1 << 16) | (1 << 12) | (1 << 14))


@wp.func
def _size_energy(held: float, boxes: float, want: float, stiffness: float) -> float:
    stretch = boxes / want - 1.0
    return stiffness * held * stretch * stretch


@wp.func
def _gain_cost(cell: int, here_potential: float, kind: wp.array(dtype=wp.int32),
               cell_energy: wp.array(dtype=wp.int64), cell_boxes: wp.array(dtype=wp.int32),
               cell_want: wp.array(dtype=wp.float32), mean_potential: wp.array(dtype=wp.float32),
               stiffness: float, push: float, push_power: float) -> float:
    """Change in H when `cell` takes one more box at a place of potential here_potential."""
    boxes = float(cell_boxes[cell])
    held = float(cell_energy[cell]) * _QUANTUM
    want = wp.max(cell_want[cell], 1.0e-6)
    cost = _size_energy(held, boxes + 1.0, want, stiffness) - _size_energy(held, boxes, want, stiffness)
    if kind[cell] == 1:
        cost -= push * wp.pow(want / (boxes + 0.5), push_power)
    else:
        cost += held * (here_potential - mean_potential[cell]) / (boxes + 1.0)
    return cost


@wp.func
def _loss_cost(cell: int, here_potential: float, kind: wp.array(dtype=wp.int32),
               cell_energy: wp.array(dtype=wp.int64), cell_boxes: wp.array(dtype=wp.int32),
               cell_want: wp.array(dtype=wp.float32), mean_potential: wp.array(dtype=wp.float32),
               stiffness: float, push: float, push_power: float) -> float:
    """Change in H when `cell` gives up its box at a place of potential here_potential."""
    boxes = float(cell_boxes[cell])
    held = float(cell_energy[cell]) * _QUANTUM
    want = wp.max(cell_want[cell], 1.0e-6)
    cost = _size_energy(held, boxes - 1.0, want, stiffness) - _size_energy(held, boxes, want, stiffness)
    if kind[cell] == 1:
        cost += push * wp.pow(want / wp.max(boxes - 0.5, 0.5), push_power)
    else:
        cost -= held * (here_potential - mean_potential[cell]) / wp.max(boxes - 1.0, 1.0)
    return cost


@wp.func
def _stays_connected(owner: wp.array3d(dtype=wp.int32), i: int, j: int, k: int, cell: int, side: int) -> int:
    """1 when the face neighbours of box (i, j, k) owned by `cell` join up in its 3x3x3 without it."""
    mask = int(0)
    for spot in range(27):
        if spot != 13:
            if box_owner(owner, i + spot // 9 - 1, j + (spot // 3) % 3 - 1, k + spot % 3 - 1, side) == cell:
                mask = mask | (1 << spot)
    faces = mask & _FACE_BITS
    if faces == 0:
        return 0
    reached = faces & (-faces)
    for sweep in range(26):
        grown = reached
        for spot in range(27):
            if ((mask >> spot) & 1) == 1 and ((grown >> spot) & 1) == 0:
                touch = 0
                if spot // 9 > 0 and ((grown >> (spot - 9)) & 1) == 1:
                    touch = 1
                if spot // 9 < 2 and ((grown >> (spot + 9)) & 1) == 1:
                    touch = 1
                if (spot // 3) % 3 > 0 and ((grown >> (spot - 3)) & 1) == 1:
                    touch = 1
                if (spot // 3) % 3 < 2 and ((grown >> (spot + 3)) & 1) == 1:
                    touch = 1
                if spot % 3 > 0 and ((grown >> (spot - 1)) & 1) == 1:
                    touch = 1
                if spot % 3 < 2 and ((grown >> (spot + 1)) & 1) == 1:
                    touch = 1
                if touch == 1:
                    grown = grown | (1 << spot)
        if (faces & grown) == faces:
            return 1
        if grown == reached:
            return 0
        reached = grown
    return 0


@wp.func
def _pair_slot(neighbour: wp.array(dtype=wp.int32), slots: int, first: int, second: int) -> int:
    """Index of the pair's face count in the contact table, -1 when the two do not touch."""
    low = wp.min(first, second)
    high = wp.max(first, second)
    for slot in range(slots):
        found = neighbour[low * slots + slot]
        if found == high:
            return low * slots + slot
        if found < 0:
            return -1
    return -1


@wp.func
def _face_cell(owner: wp.array3d(dtype=wp.int32), here: wp.vec3i, face: int, side: int) -> int:
    there = here + direction(face)
    return box_owner(owner, there[0], there[1], there[2], side)


@wp.func
def _faces_on(owner: wp.array3d(dtype=wp.int32), here: wp.vec3i, cell: int, side: int) -> int:
    """How many of box `here`'s faces border `cell`."""
    count = int(0)
    for face in range(6):
        if _face_cell(owner, here, face, side) == cell:
            count += 1
    return count


@wp.func
def _other(owner: wp.array3d(dtype=wp.int32), here: wp.vec3i, face: int, source: int, target: int,
           side: int) -> int:
    """The cell through `face` when it is neither source nor target nor empty and not seen at a lower face, else -1."""
    cell = _face_cell(owner, here, face, side)
    if cell < 0 or cell == source or cell == target:
        return -1
    for earlier in range(face):
        if _face_cell(owner, here, earlier, side) == cell:
            return -1
    return cell


@wp.func
def _restore_from(neighbour: wp.array(dtype=wp.int32), slots: int, faces: wp.array(dtype=wp.int32),
                  owner: wp.array3d(dtype=wp.int32), here: wp.vec3i, source: int, target: int, upto: int,
                  side: int):
    """Undo the source-side decrements made for faces below `upto`."""
    for face in range(upto):
        other = _other(owner, here, face, source, target, side)
        if other >= 0:
            wp.atomic_add(faces, _pair_slot(neighbour, slots, source, other), _faces_on(owner, here, other, side))


@wp.func
def _release(gained: wp.array(dtype=wp.int32), lost: wp.array(dtype=wp.int32), source: int, target: int):
    wp.atomic_add(gained, target, -1)
    if source >= 0:
        wp.atomic_add(lost, source, -1)


@wp.kernel
def _moves(owner: wp.array3d(dtype=wp.int32), energy: wp.array3d(dtype=wp.int64),
           momentum: wp.array3d(dtype=wp.vec3), potential: wp.array3d(dtype=wp.float32),
           kind: wp.array(dtype=wp.int32), cell_energy: wp.array(dtype=wp.int64),
           cell_boxes: wp.array(dtype=wp.int32), cell_want: wp.array(dtype=wp.float32),
           mean_potential: wp.array(dtype=wp.float32), cell_clock: wp.array(dtype=wp.float32),
           gained: wp.array(dtype=wp.int32), lost: wp.array(dtype=wp.int32),
           neighbour: wp.array(dtype=wp.int32), slots: int, faces: wp.array(dtype=wp.int32), offset: wp.vec3i,
           side: int,
           stiffness: float, push: float, push_power: float, move_share: float, clock_max: float, seed: int,
           moved: wp.array(dtype=wp.int32)):
    ti, tj, tk = wp.tid()
    i = 2 * ti + offset[0]
    j = 2 * tj + offset[1]
    k = 2 * tk + offset[2]
    if i >= side or j >= side or k >= side:
        return
    here = wp.vec3i(i, j, k)
    source = owner[i, j, k]
    here_potential = potential[i, j, k]
    leaving = float(0.0)
    if source >= 0:
        leaving = _loss_cost(source, here_potential, kind, cell_energy, cell_boxes, cell_want, mean_potential,
                             stiffness, push, push_power)
    target = int(-1)
    best = float(0.0)
    for face in range(6):
        cell = _face_cell(owner, here, face, side)
        if cell >= 0 and cell != source:
            change = leaving + _gain_cost(cell, here_potential, kind, cell_energy, cell_boxes, cell_want,
                                          mean_potential, stiffness, push, push_power)
            if change < best:
                best = change
                target = cell
    if target < 0:
        return
    rate = cell_clock[target]
    if source >= 0:
        rate = wp.min(rate, cell_clock[source])
    state = wp.rand_init(seed, (i * side + j) * side + k)
    if wp.randf(state) * clock_max >= rate:
        return
    for face in range(6):
        other = _other(owner, here, face, source, target, side)
        if other >= 0:
            if kind[other] == kind[target] or _pair_slot(neighbour, slots, target, other) < 0:
                return
    if source >= 0:
        if cell_boxes[source] <= 1 or _stays_connected(owner, i, j, k, source, side) == 0:
            return
    target_cap = wp.max(1, int(float(cell_boxes[target]) * move_share))
    if wp.atomic_add(gained, target, 1) >= target_cap:
        wp.atomic_add(gained, target, -1)
        return
    if source >= 0:
        source_cap = wp.max(1, int(float(cell_boxes[source]) * move_share))
        if wp.atomic_add(lost, source, 1) >= source_cap:
            wp.atomic_add(lost, source, -1)
            wp.atomic_add(gained, target, -1)
            return
        failed = int(6)
        for face in range(6):
            other = _other(owner, here, face, source, target, side)
            if other >= 0 and failed == 6:
                count = _faces_on(owner, here, other, side)
                slot = _pair_slot(neighbour, slots, source, other)
                if wp.atomic_add(faces, slot, -count) - count < 1:
                    wp.atomic_add(faces, slot, count)
                    failed = face
        if failed < 6:
            _restore_from(neighbour, slots, faces, owner, here, source, target, failed, side)
            _release(gained, lost, source, target)
            return
        net = _faces_on(owner, here, source, side) - _faces_on(owner, here, target, side)
        shared = _pair_slot(neighbour, slots, source, target)
        if net < 0:
            if wp.atomic_add(faces, shared, net) + net < 1:
                wp.atomic_add(faces, shared, -net)
                _restore_from(neighbour, slots, faces, owner, here, source, target, 6, side)
                _release(gained, lost, source, target)
                return
        if net > 0:
            wp.atomic_add(faces, shared, net)
    for face in range(6):
        other = _other(owner, here, face, source, target, side)
        if other >= 0:
            wp.atomic_add(faces, _pair_slot(neighbour, slots, target, other), _faces_on(owner, here, other, side))
    if source >= 0:
        for face in range(6):
            if _face_cell(owner, here, face, side) == source:
                there = here + direction(face)
                wp.atomic_add(energy, there[0], there[1], there[2], energy[i, j, k])
                wp.atomic_add(momentum, there[0], there[1], there[2], momentum[i, j, k])
                energy[i, j, k] = wp.int64(0)
                momentum[i, j, k] = wp.vec3(0.0, 0.0, 0.0)
                break
        wp.atomic_add(cell_boxes, source, -1)
    for face in range(6):
        if _face_cell(owner, here, face, side) == target:
            there = here + direction(face)
            given = energy[there[0], there[1], there[2]] / wp.int64(3)
            if given > wp.int64(0):
                share = float(wp.float64(given) / wp.float64(energy[there[0], there[1], there[2]]))
                carried = momentum[there[0], there[1], there[2]] * share
                wp.atomic_add(energy, there[0], there[1], there[2], -given)
                wp.atomic_add(momentum, there[0], there[1], there[2], -carried)
                energy[i, j, k] = given
                momentum[i, j, k] = carried
            break
    owner[i, j, k] = target
    wp.atomic_add(cell_boxes, target, 1)
    wp.atomic_add(moved, 0, 1)


def box_moves(universe: Universe, settings: Settings) -> int:
    """One sweep of eight parity phases of border moves; the number of boxes that changed owner."""
    device = universe.owner.device
    moved = wp.zeros(1, dtype=wp.int32, device=device)
    half = (universe.side + 1) // 2
    clock_max = (settings.clock_scale + 1.0) / settings.clock_scale
    for phase in range(8):
        universe.gained.zero_()
        universe.lost.zero_()
        offset = wp.vec3i(phase // 4, (phase // 2) % 2, phase % 2)
        seed = (universe.seed * 1000003 + universe.steps * 8 + phase) % (2 ** 31 - 1)
        wp.launch(_moves, dim=(half, half, half), device=device,
                  inputs=[universe.owner, universe.energy, universe.momentum, universe.potential, universe.kind,
                          universe.cell_energy, universe.cell_boxes, universe.cell_want,
                          universe.cell_potential_mean, universe.cell_clock, universe.gained, universe.lost,
                          universe.neighbour, universe.slots, universe.faces, offset, universe.side, settings.stiffness,
                          settings.push, settings.push_power, settings.move_share, clock_max, seed, moved])
    return int(moved.numpy()[0])
