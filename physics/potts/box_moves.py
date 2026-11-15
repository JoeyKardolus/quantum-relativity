"""Box moves: cells change size and shape by plate moves, a cell border shifting by one box.

Contract: `box_moves(universe, settings, correcting=False)` runs one sweep of 64 phases
on the GPU and returns the number of boxes that changed owner (one small copy to the
host). A phase visits the sites b of the active region (active_region.py) whose
coordinates are one pattern mod 4; a site's whole work stays inside the 4x4x4 block
from b - 1 to b + 2, and those blocks never overlap in a phase, so moves never race on
boxes (only on per-cell counters, which are atomic).
A plate is 1x1, 1x2, 2x1 or 2x2 boxes with its corner at b, flat across one axis a; a
plate move gives every plate box the owner of its neighbour one box along +a or -a:
the border shifts by one box there. A single box cannot move a place where four cells
meet along an edge (A, B, A, B round it), nor where eight meet at a corner, without
making like touch like; a plate across that edge or corner moves it whole, so a cell's
edges and corners move like its faces (the earlier one-box moves froze every inner edge
and corner on the start lattice). Each site tries settings.proposals random plates
(3 axes x 2 senses x 4 shapes) and makes the first that passes:
  1. Cost. The change in
     H = sum over cells of stiffness * E * ln(n / aim)**2      (size follows energy;
         aim = want for an energy volume, its swing for a joint, joint_swing.py; in
         ln(n / aim), so a cell far from its want presses as hard as length = c / E
         says, scale-free: with (n / aim - 1)**2 a cell wanting 64 times its boxes
         could not take one from a neighbour at its want, measured 2026-11-15)
       + surface * (E / n) * (its box faces bordering another cell or empty space)
         (the border energy of the cellular Potts method the bible names, Graner and
         Glazier 1992: without it cells grew into sponges, measured 2026-11-15, a cell of
         2,322 boxes spread over 34 x 37 x 23)
       - for joints, the work of their push P = push * (want / n)**push_power
       + for energy volumes, E * (mean potential over its boxes)     (the pull)
       + over every link of a changed cell, link_stiffness * mean E * (length / rest
         - 1)**2 where the link, the distance between the two cells' centres, is longer
         than its rest, half of each cell's width n**(1/3): links bend, never stretch
     must be below 0 (n, E, want, aim, potential and centres are the cells', kept live).
     The link term is weighed only for a plate that already lowers the rest.
  2. Time. It then happens with probability min over the changed cells of their clock,
     over clock_max = (s + 1) / s (bible: time is local).
  3. Rules. Every face a changed box shares with another cell must join cells of
     unlike kinds that already touch (ABAB; no new contact); no contact's face count
     may fall to 0 (no contact ends), counts changed atomically and undone whole on
     failure; every cell losing boxes must stay one piece (its boxes next to the lost
     ones join up inside the block, at least one of them outside the plate) and keep
     at least one box; a cell may gain, and
     lose, at most max(4, move_share * n) boxes per sweep.
  4. Energy. A box's energy, momentum and heat go, exactly, to a box of its old cell
     in the block; a box joining a cell takes half of what its neighbour along a holds
     (no other site touches that box), so a cell stretches into a box rather than
     leaving it a vacuum. A move changes only sizes and places, never a cell's energy.
`correcting=True` is the start's size correction (size_correction.py): only the size
and border terms, aimed at the want, no clock, no move limit.
A box never becomes empty: space only grows, as cells claim empty edge boxes inside
the active region. Atomics make which of two racing moves wins, and so a run, not
exactly repeatable.
"""
from __future__ import annotations

import warp as wp

from physics.potts.box_owner import box_owner
from physics.potts.direction import direction
from physics.potts.settings import Settings
from physics.potts.universe import QUANTUM, Universe

_QUANTUM = wp.constant(QUANTUM)
_RU0 = wp.constant(wp.uint64(0x111111111111))
_RU3 = wp.constant(wp.uint64(0x888888888888))
_RV0 = wp.constant(wp.uint64(0x000F000F000F))
_RV3 = wp.constant(wp.uint64(0xF000F000F000))
_FULL = wp.constant(wp.uint64(0xFFFFFFFFFFFF))
_vec8i = wp.types.vector(length=8, dtype=wp.int32)
_vec8f = wp.types.vector(length=8, dtype=wp.float32)
_vec24i = wp.types.vector(length=24, dtype=wp.int32)


@wp.func
def _plate_index(place: wp.vec3i, corner: wp.vec3i, axis: int, wide_u: int, wide_v: int) -> int:
    """The plate index 0..3 of box `place`, or -1 when it is not a plate box."""
    offset = place - corner
    along_u = offset[(axis + 1) % 3]
    along_v = offset[(axis + 2) % 3]
    if offset[axis] != 0 or along_u < 0 or along_v < 0 or along_u >= wide_u or along_v >= wide_v:
        return -1
    return along_u + 2 * along_v


@wp.func
def _after(owner: wp.array3d(dtype=wp.int32), place: wp.vec3i, corner: wp.vec3i, axis: int, wide_u: int,
           wide_v: int, new: wp.vec4i, side: int) -> int:
    """The owner of box `place` once the plate has moved."""
    index = _plate_index(place, corner, axis, wide_u, wide_v)
    if index >= 0:
        return new[index]
    return box_owner(owner, place[0], place[1], place[2], side)


@wp.func
def _before(owner: wp.array3d(dtype=wp.int32), place: wp.vec3i, corner: wp.vec3i, axis: int, wide_u: int,
            wide_v: int, old: wp.vec4i, side: int) -> int:
    """The owner of box `place` before the plate moves."""
    index = _plate_index(place, corner, axis, wide_u, wide_v)
    if index >= 0:
        return old[index]
    return box_owner(owner, place[0], place[1], place[2], side)


@wp.func
def _plate_box(corner: wp.vec3i, axis: int, index: int) -> wp.vec3i:
    return corner + direction(2 * ((axis + 1) % 3)) * (index % 2) + direction(2 * ((axis + 2) % 3)) * (index // 2)


@wp.func
def _block_box(corner: wp.vec3i, axis: int, bit: int) -> wp.vec3i:
    """The box of bit 0..47 of the block around the plate: bit = u + 4 v + 16 a, each from -1."""
    return (corner + direction(2 * ((axis + 1) % 3)) * (bit % 4 - 1) + direction(2 * ((axis + 2) % 3)) * (
        (bit // 4) % 4 - 1) + direction(2 * axis) * (bit // 16 - 1))


@wp.func
def _block_bit(place: wp.vec3i, corner: wp.vec3i, axis: int) -> int:
    offset = place - corner
    return (offset[(axis + 1) % 3] + 1) + 4 * (offset[(axis + 2) % 3] + 1) + 16 * (offset[axis] + 1)


@wp.func
def _size_energy(held: float, boxes: float, aim: float, stiffness: float) -> float:
    stretch = wp.log(wp.max(boxes, 0.5) / aim)
    return stiffness * held * stretch * stretch


@wp.func
def _box_energy(cell_energy: wp.array(dtype=wp.int64), cell_boxes: wp.array(dtype=wp.int32), cell: int) -> float:
    """A cell's energy per box; 0 for empty space (-1), which has no border energy."""
    if cell < 0:
        return 0.0
    return float(cell_energy[cell]) * _QUANTUM / float(wp.max(cell_boxes[cell], 1))


@wp.func
def _slot_of(cells: _vec8i, count: int, cell: int) -> int:
    for slot in range(8):
        if slot < count and cells[slot] == cell:
            return slot
    return -1


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
def _centre(cell_place: wp.array(dtype=wp.int64), cell: int, shift: wp.vec3, boxes: float) -> wp.vec3:
    total = wp.vec3(float(cell_place[3 * cell]), float(cell_place[3 * cell + 1]), float(cell_place[3 * cell + 2]))
    return (total + shift) / wp.max(boxes, 1.0)


@wp.func
def _stretch(here: wp.vec3, there: wp.vec3, boxes: float, other_boxes: float) -> float:
    rest = 0.5 * (wp.pow(wp.max(boxes, 1.0), 1.0 / 3.0) + wp.pow(wp.max(other_boxes, 1.0), 1.0 / 3.0))
    return wp.max(0.0, wp.length(here - there) / rest - 1.0)


@wp.func
def _joined(mask: wp.uint64, seeds: wp.uint64) -> int:
    """1 when every seed bit is reached from the lowest one through face steps inside mask."""
    if seeds == wp.uint64(0):
        return 0
    reached = seeds & (~seeds + wp.uint64(1))
    for sweep in range(48):
        grown = reached | ((reached << wp.uint64(1)) & ~_RU0) | ((reached >> wp.uint64(1)) & ~_RU3)
        grown = grown | ((reached << wp.uint64(4)) & ~_RV0) | ((reached >> wp.uint64(4)) & ~_RV3)
        grown = grown | (reached << wp.uint64(16)) | (reached >> wp.uint64(16))
        grown = grown & mask & _FULL
        if (grown & seeds) == seeds:
            return 1
        if grown == reached:
            return 0
        reached = grown
    return 0


@wp.func
def _net(old_low: _vec24i, old_high: _vec24i, new_low: _vec24i, new_high: _vec24i, entry: int) -> int:
    """The net face-count change of entry's pair (entries 0..23 old pairs, 24..47 new), 0 unless first seen."""
    low = int(0)
    high = int(0)
    if entry < 24:
        low = old_low[entry]
        high = old_high[entry]
    else:
        low = new_low[entry - 24]
        high = new_high[entry - 24]
    if low < 0:
        return 0
    for earlier in range(48):
        if earlier < entry:
            if earlier < 24:
                if old_low[earlier] == low and old_high[earlier] == high:
                    return 0
            else:
                if new_low[earlier - 24] == low and new_high[earlier - 24] == high:
                    return 0
    net = int(0)
    for record in range(24):
        if new_low[record] == low and new_high[record] == high:
            net += 1
        if old_low[record] == low and old_high[record] == high:
            net -= 1
    return net


@wp.func
def _entry_pair(old_low: _vec24i, old_high: _vec24i, new_low: _vec24i, new_high: _vec24i, entry: int) -> wp.vec2i:
    if entry < 24:
        return wp.vec2i(old_low[entry], old_high[entry])
    return wp.vec2i(new_low[entry - 24], new_high[entry - 24])


@wp.kernel
def _moves(owner: wp.array3d(dtype=wp.int32), energy: wp.array3d(dtype=wp.int64),
           momentum: wp.array3d(dtype=wp.vec3), heat: wp.array3d(dtype=wp.float32),
           potential: wp.array3d(dtype=wp.float32), kind: wp.array(dtype=wp.int32),
           cell_energy: wp.array(dtype=wp.int64), cell_boxes: wp.array(dtype=wp.int32),
           cell_want: wp.array(dtype=wp.float32), cell_aim: wp.array(dtype=wp.float32),
           mean_potential: wp.array(dtype=wp.float32), cell_clock: wp.array(dtype=wp.float32),
           cell_place: wp.array(dtype=wp.int64), gained: wp.array(dtype=wp.int32), lost: wp.array(dtype=wp.int32),
           neighbour: wp.array(dtype=wp.int32), slots: int, faces: wp.array(dtype=wp.int32),
           links: wp.array(dtype=wp.int32), link_slots: int, offset: wp.vec3i, low: wp.vec3i, high: wp.vec3i,
           side: int, stiffness: float, surface: float, push: float, push_power: float, link_stiffness: float,
           move_share: float,
           clock_max: float, proposals: int, correcting: int, seed: int, moved: wp.array(dtype=wp.int32)):
    ti, tj, tk = wp.tid()
    corner = wp.vec3i(low[0] + 4 * ti + offset[0], low[1] + 4 * tj + offset[1], low[2] + 4 * tk + offset[2])
    if corner[0] >= high[0] or corner[1] >= high[1] or corner[2] >= high[2]:
        return
    state = wp.rand_init(seed, (corner[0] * side + corner[1]) * side + corner[2])
    timed = int(0)
    for attempt in range(proposals):
        choice = wp.randi(state, 0, 24)
        axis = choice % 3
        sense = 1 - 2 * ((choice // 3) % 2)
        wide_u = 1 + (choice // 6) % 2
        wide_v = 1 + (choice // 12)
        along = direction(2 * axis) * sense
        old = wp.vec4i(-2, -2, -2, -2)
        new = wp.vec4i(-2, -2, -2, -2)
        changed = int(0)
        usable = int(1)
        for index in range(4):
            if index % 2 < wide_u and index // 2 < wide_v:
                place = _plate_box(corner, axis, index)
                if (place[0] >= high[0] or place[1] >= high[1] or place[2] >= high[2]
                        or place[0] < low[0] or place[1] < low[1] or place[2] < low[2]):
                    usable = 0
                else:
                    source = place + along
                    old[index] = owner[place[0], place[1], place[2]]
                    new[index] = box_owner(owner, source[0], source[1], source[2], side)
                    if new[index] < 0:
                        usable = 0
                    elif new[index] != old[index]:
                        changed += 1
        if usable == 0 or changed == 0:
            continue
        surface_change = float(0.0)
        old_low = _vec24i()
        old_high = _vec24i()
        new_low = _vec24i()
        new_high = _vec24i()
        for record in range(24):
            old_low[record] = -1
            old_high[record] = -1
            new_low[record] = -1
            new_high[record] = -1
            index = record // 6
            face = record % 6
            if old[index] != -2 and new[index] != old[index]:
                place = _plate_box(corner, axis, index)
                there = place + direction(face)
                other_index = _plate_index(there, corner, axis, wide_u, wide_v)
                counted = int(1)
                if other_index >= 0 and other_index < index:
                    if new[other_index] != old[other_index]:
                        counted = 0
                if counted == 1:
                    was_cell = _before(owner, there, corner, axis, wide_u, wide_v, old, side)
                    now_cell = _after(owner, there, corner, axis, wide_u, wide_v, new, side)
                    if was_cell != old[index]:
                        surface_change -= _box_energy(cell_energy, cell_boxes, was_cell) + _box_energy(
                            cell_energy, cell_boxes, old[index])
                    if now_cell != new[index]:
                        surface_change += _box_energy(cell_energy, cell_boxes, now_cell) + _box_energy(
                            cell_energy, cell_boxes, new[index])
                    if was_cell >= 0 and old[index] >= 0 and was_cell != old[index]:
                        old_low[record] = wp.min(was_cell, old[index])
                        old_high[record] = wp.max(was_cell, old[index])
                    if now_cell >= 0 and now_cell != new[index]:
                        new_low[record] = wp.min(now_cell, new[index])
                        new_high[record] = wp.max(now_cell, new[index])
        cells = _vec8i(-1, -1, -1, -1, -1, -1, -1, -1)
        delta = _vec8i(0, 0, 0, 0, 0, 0, 0, 0)
        loses = _vec8i(0, 0, 0, 0, 0, 0, 0, 0)
        pulled = _vec8f(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        shift_x = _vec8f(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        shift_y = _vec8f(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        shift_z = _vec8f(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        count = int(0)
        for index in range(4):
            if old[index] != -2 and new[index] != old[index]:
                place = _plate_box(corner, axis, index)
                here = potential[place[0], place[1], place[2]]
                for role in range(2):
                    cell = new[index]
                    sign = int(1)
                    if role == 0:
                        cell = old[index]
                        sign = -1
                    if cell >= 0:
                        slot = _slot_of(cells, count, cell)
                        if slot < 0:
                            slot = count
                            cells[slot] = cell
                            count += 1
                        delta[slot] = delta[slot] + sign
                        pulled[slot] = pulled[slot] + float(sign) * here
                        shift_x[slot] = shift_x[slot] + float(sign * place[0])
                        shift_y[slot] = shift_y[slot] + float(sign * place[1])
                        shift_z[slot] = shift_z[slot] + float(sign * place[2])
                        if role == 0:
                            loses[slot] = 1
        total = surface * surface_change
        rate = clock_max
        viable = int(1)
        for slot in range(8):
            if slot < count:
                cell = cells[slot]
                boxes = float(cell_boxes[cell])
                after = boxes + float(delta[slot])
                if after < 1.0:
                    viable = 0
                held = float(cell_energy[cell]) * _QUANTUM
                want = wp.max(cell_want[cell], 1.0e-6)
                aim = want
                if correcting == 0:
                    aim = wp.max(cell_aim[cell], 1.0e-6)
                total += _size_energy(held, after, aim, stiffness) - _size_energy(held, boxes, aim, stiffness)
                if correcting == 0:
                    if kind[cell] == 1:
                        total -= push * wp.pow(want / wp.max(boxes + 0.5 * float(delta[slot]), 0.5),
                                               push_power) * float(delta[slot])
                    else:
                        mean = mean_potential[cell]
                        total += held * ((mean * boxes + pulled[slot]) / wp.max(after, 1.0) - mean)
                rate = wp.min(rate, cell_clock[cell])
        if viable == 0 or total >= 0.0:
            continue
        if correcting == 0 and link_stiffness > 0.0:
            for slot in range(8):
                if slot < count:
                    cell = cells[slot]
                    boxes = float(cell_boxes[cell])
                    after = boxes + float(delta[slot])
                    held = float(cell_energy[cell]) * _QUANTUM
                    was = _centre(cell_place, cell, wp.vec3(0.0, 0.0, 0.0), boxes)
                    now = _centre(cell_place, cell, wp.vec3(shift_x[slot], shift_y[slot], shift_z[slot]), after)
                    for link in range(link_slots):
                        other = links[cell * link_slots + link]
                        if other < 0:
                            break
                        other_boxes = float(cell_boxes[other])
                        other_after = other_boxes
                        weight = link_stiffness * 0.5 * (held + float(cell_energy[other]) * _QUANTUM)
                        other_was = _centre(cell_place, other, wp.vec3(0.0, 0.0, 0.0), other_boxes)
                        other_now = other_was
                        partner = _slot_of(cells, count, other)
                        if partner >= 0:
                            other_after = other_boxes + float(delta[partner])
                            other_now = _centre(cell_place, other, wp.vec3(shift_x[partner], shift_y[partner],
                                                                           shift_z[partner]), other_after)
                            weight = 0.5 * weight
                        before = _stretch(was, other_was, boxes, other_boxes)
                        later = _stretch(now, other_now, after, other_after)
                        total += weight * (later * later - before * before)
            if total >= 0.0:
                continue
        if correcting == 0 and timed == 0:
            timed = 1
            if wp.randf(state) * clock_max >= rate:
                return
        legal = int(1)
        for index in range(4):
            if old[index] != -2 and new[index] != old[index]:
                place = _plate_box(corner, axis, index)
                target = new[index]
                for face in range(6):
                    there = place + direction(face)
                    other = _after(owner, there, corner, axis, wide_u, wide_v, new, side)
                    if other >= 0 and other != target:
                        if kind[other] == kind[target] or _pair_slot(neighbour, slots, target, other) < 0:
                            legal = 0
        if legal == 0:
            continue
        for slot in range(8):
            if slot < count and loses[slot] == 1 and legal == 1:
                cell = cells[slot]
                mask = wp.uint64(0)
                for bit in range(48):
                    if _after(owner, _block_box(corner, axis, bit), corner, axis, wide_u, wide_v, new, side) == cell:
                        mask = mask | (wp.uint64(1) << wp.uint64(bit))
                seeds = wp.uint64(0)
                outside = int(0)
                for index in range(4):
                    if old[index] == cell and new[index] != cell:
                        place = _plate_box(corner, axis, index)
                        for face in range(6):
                            there = place + direction(face)
                            if _after(owner, there, corner, axis, wide_u, wide_v, new, side) == cell:
                                seeds = seeds | (wp.uint64(1) << wp.uint64(_block_bit(there, corner, axis)))
                                if _plate_index(there, corner, axis, wide_u, wide_v) < 0:
                                    outside = 1
                if outside == 0 or _joined(mask, seeds) == 0:
                    legal = 0
        if legal == 0:
            continue
        failed = int(8)
        if correcting == 0:
            for slot in range(8):
                if slot < count and failed == 8:
                    cell = cells[slot]
                    cap = wp.max(4, int(float(cell_boxes[cell]) * move_share))
                    change = delta[slot]
                    if change > 0:
                        if wp.atomic_add(gained, cell, change) + change > cap:
                            failed = slot
                    elif change < 0:
                        if wp.atomic_add(lost, cell, -change) - change > cap:
                            failed = slot
            if failed < 8:
                for slot in range(8):
                    if slot <= failed:
                        change = delta[slot]
                        if change > 0:
                            wp.atomic_add(gained, cells[slot], -change)
                        elif change < 0:
                            wp.atomic_add(lost, cells[slot], change)
                return
        broken = int(48)
        for entry in range(48):
            if broken == 48:
                net = _net(old_low, old_high, new_low, new_high, entry)
                if net < 0:
                    pair = _entry_pair(old_low, old_high, new_low, new_high, entry)
                    slot = _pair_slot(neighbour, slots, pair[0], pair[1])
                    if slot < 0:
                        broken = entry
                    elif wp.atomic_add(faces, slot, net) + net < 1:
                        broken = entry
        if broken < 48:
            for entry in range(48):
                if entry <= broken:
                    net = _net(old_low, old_high, new_low, new_high, entry)
                    if net < 0:
                        pair = _entry_pair(old_low, old_high, new_low, new_high, entry)
                        slot = _pair_slot(neighbour, slots, pair[0], pair[1])
                        if slot >= 0:
                            wp.atomic_add(faces, slot, -net)
            if correcting == 0:
                for slot in range(8):
                    if slot < count:
                        change = delta[slot]
                        if change > 0:
                            wp.atomic_add(gained, cells[slot], -change)
                        elif change < 0:
                            wp.atomic_add(lost, cells[slot], change)
            return
        for entry in range(48):
            net = _net(old_low, old_high, new_low, new_high, entry)
            if net > 0:
                pair = _entry_pair(old_low, old_high, new_low, new_high, entry)
                wp.atomic_add(faces, _pair_slot(neighbour, slots, pair[0], pair[1]), net)
        for index in range(4):
            if old[index] >= 0 and new[index] != old[index]:
                place = _plate_box(corner, axis, index)
                cell = old[index]
                home = wp.vec3i(-1, -1, -1)
                for face in range(6):
                    there = place + direction(face)
                    if home[0] < 0 and _plate_index(there, corner, axis, wide_u, wide_v) < 0:
                        if box_owner(owner, there[0], there[1], there[2], side) == cell:
                            home = there
                if home[0] < 0:
                    for bit in range(48):
                        there = _block_box(corner, axis, bit)
                        if home[0] < 0 and _plate_index(there, corner, axis, wide_u, wide_v) < 0:
                            if box_owner(owner, there[0], there[1], there[2], side) == cell:
                                home = there
                energy[home[0], home[1], home[2]] = energy[home[0], home[1], home[2]] + energy[place[0], place[1],
                                                                                               place[2]]
                momentum[home[0], home[1], home[2]] = momentum[home[0], home[1], home[2]] + momentum[
                    place[0], place[1], place[2]]
                heat[home[0], home[1], home[2]] = heat[home[0], home[1], home[2]] + heat[place[0], place[1], place[2]]
                energy[place[0], place[1], place[2]] = wp.int64(0)
                momentum[place[0], place[1], place[2]] = wp.vec3(0.0, 0.0, 0.0)
                heat[place[0], place[1], place[2]] = 0.0
                wp.atomic_add(cell_boxes, cell, -1)
                wp.atomic_add(cell_place, 3 * cell, wp.int64(-place[0]))
                wp.atomic_add(cell_place, 3 * cell + 1, wp.int64(-place[1]))
                wp.atomic_add(cell_place, 3 * cell + 2, wp.int64(-place[2]))
        for index in range(4):
            if old[index] != -2 and new[index] != old[index]:
                place = _plate_box(corner, axis, index)
                source = place + along
                cell = new[index]
                given = energy[source[0], source[1], source[2]] / wp.int64(2)
                fraction = float(0.0)
                if energy[source[0], source[1], source[2]] > wp.int64(0):
                    fraction = float(wp.float64(given) / wp.float64(energy[source[0], source[1], source[2]]))
                carried = momentum[source[0], source[1], source[2]] * fraction
                warmth = heat[source[0], source[1], source[2]] * fraction
                energy[source[0], source[1], source[2]] = energy[source[0], source[1], source[2]] - given
                momentum[source[0], source[1], source[2]] = momentum[source[0], source[1], source[2]] - carried
                heat[source[0], source[1], source[2]] = heat[source[0], source[1], source[2]] - warmth
                energy[place[0], place[1], place[2]] = given
                momentum[place[0], place[1], place[2]] = carried
                heat[place[0], place[1], place[2]] = warmth
                wp.atomic_add(cell_boxes, cell, 1)
                wp.atomic_add(cell_place, 3 * cell, wp.int64(place[0]))
                wp.atomic_add(cell_place, 3 * cell + 1, wp.int64(place[1]))
                wp.atomic_add(cell_place, 3 * cell + 2, wp.int64(place[2]))
        for index in range(4):
            if old[index] != -2 and new[index] != old[index]:
                place = _plate_box(corner, axis, index)
                owner[place[0], place[1], place[2]] = new[index]
        wp.atomic_add(moved, 0, changed)
        return


def box_moves(universe: Universe, settings: Settings, correcting: bool = False) -> int:
    """One sweep of 64 phases of plate moves; the number of boxes that changed owner."""
    device = universe.owner.device
    moved = wp.zeros(1, dtype=wp.int32, device=device)
    dims = tuple(-(-(top - bottom) // 4) for bottom, top in zip(universe.low, universe.high))
    clock_max = (settings.clock_scale + 1.0) / settings.clock_scale
    universe.gained.zero_()
    universe.lost.zero_()
    sweep = universe.scratch.get("sweeps", 0)
    universe.scratch["sweeps"] = sweep + 1
    for phase in range(64):
        offset = wp.vec3i(phase // 16, (phase // 4) % 4, phase % 4)
        seed = (universe.seed * 1000003 + sweep * 64 + phase) % (2 ** 31 - 1)
        wp.launch(_moves, dim=dims, device=device,
                  inputs=[universe.owner, universe.energy, universe.momentum, universe.heat, universe.potential,
                          universe.kind, universe.cell_energy, universe.cell_boxes, universe.cell_want,
                          universe.cell_aim, universe.cell_potential_mean, universe.cell_clock, universe.cell_place,
                          universe.gained, universe.lost, universe.neighbour, universe.slots, universe.faces,
                          universe.links, universe.link_slots, offset, wp.vec3i(*universe.low),
                          wp.vec3i(*universe.high), universe.side, settings.stiffness, settings.surface, settings.push,
                          settings.push_power, settings.link_stiffness, settings.move_share, clock_max,
                          int(settings.proposals), int(correcting), seed, moved])
    return int(moved.numpy()[0])
