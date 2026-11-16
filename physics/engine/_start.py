"""Start kernels: the per-box work of new_universe.py on the GPU, so a start costs the host only its uneven field.

Helpers of new_universe.py (its contract says what each step means):
  held_boxes    each volume's plus boxes after the carving, counted exactly
  face_partner  each joint box's partner volume made a lattice face neighbour of its volume
  cell_ids      each filled box's cell id written into the state's owner array; each cell's
                box count and the room its boxes lie in (round the edge when wrapping)
  fill_cells    each cell's whole quanta shared over its boxes, largest remainders first
Every sum that decides a whole quantum runs in one fixed order (a cell's boxes in rising
box order), in float64, with the same operations new_universe ran on the host before
2026-11-17, so a seed gives the same universe to the last quantum.
"""
from __future__ import annotations

import warp as wp

wp.set_module_options({"fuse_fp": False})

PLUS = -2
OUTER = -1


@wp.func
def grid_of(volume: int, count: int) -> wp.vec3i:
    """A volume's lattice place (three indices 0 .. count - 1) from its index."""
    return wp.vec3i(volume // (count * count), (volume // count) % count, volume % count)


@wp.func
def _apart(first: wp.vec3i, second: wp.vec3i, count: int, wrap: int) -> wp.vec3i:
    """second - first on the lattice, the short way round when the universe wraps (more than 2 volumes a side)."""
    apart = second - first
    if wrap == 1 and count > 2:
        for axis in range(3):
            apart[axis] = (apart[axis] + count // 2 + count) % count - count // 2
    return apart


@wp.kernel
def held_boxes(raw: wp.array3d(dtype=wp.int32), partner: wp.array3d(dtype=wp.int32), held: wp.array(dtype=wp.int32)):
    ti, tj, tk = wp.tid()
    if partner[ti, tj, tk] == PLUS:
        wp.atomic_add(held, raw[ti, tj, tk], 1)


@wp.kernel
def face_partner(raw: wp.array3d(dtype=wp.int32), count: int, period: wp.float64, origin: wp.float64, filled: int,
                 wrap: int, partner: wp.array3d(dtype=wp.int32), outer: wp.array(dtype=wp.int32)):
    ti, tj, tk = wp.tid()
    lower = partner[ti, tj, tk]
    if lower == PLUS:
        return
    own = raw[ti, tj, tk]
    if lower >= 0:
        apart = _apart(grid_of(own, count), grid_of(lower, count), count, wrap)
        if wp.abs(apart[0]) + wp.abs(apart[1]) + wp.abs(apart[2]) != 1:
            lower = _nearest(wp.vec3i(ti, tj, tk), grid_of(own, count), count, period, origin, filled, wrap)
            partner[ti, tj, tk] = lower
    if lower == OUTER:
        wp.atomic_max(outer, own, 1)


@wp.func
def _nearest(place: wp.vec3i, home: wp.vec3i, count: int, period: wp.float64, origin: wp.float64, filled: int,
             wrap: int) -> int:
    """The lattice face neighbour of `home` whose block centre is nearest the box (first found on a tie), -1 when none."""
    best = int(-1)
    nearest = wp.float64(1.0e300)
    for axis in range(3):
        for way in range(2):
            neighbour = home
            neighbour[axis] = neighbour[axis] + 2 * way - 1
            if wrap == 1:
                neighbour[axis] = (neighbour[axis] + count) % count
            if neighbour[axis] >= 0 and neighbour[axis] < count:
                distance = wp.float64(0.0)
                for across in range(3):
                    gap = wp.float64(float(place[across])) - (wp.float64(float(neighbour[across])) * period + origin)
                    if wrap == 1:
                        gap = gap - wp.float64(float(filled)) * wp.rint(gap / wp.float64(float(filled)))
                    distance = distance + gap * gap
                if distance < nearest:
                    nearest = distance
                    best = (neighbour[0] * count + neighbour[1]) * count + neighbour[2]
    return best


@wp.kernel
def cell_ids(raw: wp.array3d(dtype=wp.int32), partner: wp.array3d(dtype=wp.int32), count: int, extent: int,
             cells_per_side: int, filled: int, wrap: int, outer_id: wp.array(dtype=wp.int32), first: int, low: int,
             owner: wp.array3d(dtype=wp.int32), boxes: wp.array(dtype=wp.int32), spans: wp.array2d(dtype=wp.int32)):
    ti, tj, tk = wp.tid()
    own = raw[ti, tj, tk]
    other = partner[ti, tj, tk]
    home = grid_of(own, count)
    place = 2 * home + wp.vec3i(2, 2, 2)
    cell = int(0)
    if other == OUTER:
        cell = outer_id[own]
    else:
        if other >= 0:
            place = place + _apart(home, grid_of(other, count), count, wrap)
            if wrap == 1:
                for axis in range(3):
                    place[axis] = (place[axis] - 1 + cells_per_side) % cells_per_side + 1
        cell = (place[0] * extent + place[1]) * extent + place[2]
    owner[first + low + ti, low + tj, low + tk] = cell
    wp.atomic_add(boxes, cell, 1)
    here = wp.vec3i(ti, tj, tk)
    for axis in range(3):
        moved = (here[axis] + filled // 2) % filled
        wp.atomic_min(spans, cell, 4 * axis, here[axis])
        wp.atomic_max(spans, cell, 4 * axis + 1, here[axis])
        wp.atomic_min(spans, cell, 4 * axis + 2, moved)
        wp.atomic_max(spans, cell, 4 * axis + 3, moved)


@wp.func
def _along(index: int, low: int, length: int, filled: int) -> int:
    """Coordinate number `index` of a room of `length` boxes from `low` round the edge, in rising coordinate."""
    over = low + length - filled
    if over <= 0:
        return low + index
    if index < over:
        return index
    return low + index - over


@wp.func
def _room(spans: wp.array2d(dtype=wp.int32), cell: int, filled: int, wrap: int):
    """Per axis the first coordinate and the length of the room a cell's boxes lie in, round the edge when shorter."""
    lows = wp.vec3i(0, 0, 0)
    lengths = wp.vec3i(0, 0, 0)
    for axis in range(3):
        lows[axis] = spans[cell, 4 * axis]
        lengths[axis] = spans[cell, 4 * axis + 1] - spans[cell, 4 * axis] + 1
        moved = spans[cell, 4 * axis + 3] - spans[cell, 4 * axis + 2] + 1
        if wrap == 1 and moved < lengths[axis]:
            lows[axis] = (spans[cell, 4 * axis + 2] - filled // 2 + filled) % filled
            lengths[axis] = moved
    return lows, lengths


@wp.func
def _spot(index: int, lows: wp.vec3i, lengths: wp.vec3i, filled: int) -> wp.vec3i:
    """Box number `index` of a room in rising box order (first axis slowest)."""
    return wp.vec3i(_along(index // (lengths[1] * lengths[2]), lows[0], lengths[0], filled),
                    _along((index // lengths[2]) % lengths[1], lows[1], lengths[1], filled),
                    _along(index % lengths[2], lows[2], lengths[2], filled))


@wp.kernel
def fill_cells(kinds: wp.array(dtype=wp.int32), totals: wp.array(dtype=wp.float64),
               spans: wp.array2d(dtype=wp.int32), filled: int, wrap: int, first: int, low: int,
               owner: wp.array3d(dtype=wp.int32), field: wp.array3d(dtype=wp.float64),
               quanta: wp.array3d(dtype=wp.int32)):
    cell = wp.tid()
    kind = kinds[cell]
    if kind == 0:
        return
    lows, lengths = _room(spans, cell, filled, wrap)
    room = lengths[0] * lengths[1] * lengths[2]
    total = totals[cell]
    weight = wp.float64(0.0)
    members = int(0)
    for index in range(room):
        spot = _spot(index, lows, lengths, filled)
        if owner[first + low + spot[0], low + spot[1], low + spot[2]] == cell:
            members += 1
            if kind == 1:
                weight = weight + field[spot[0], spot[1], spot[2]]
    if kind == -1:
        weight = wp.float64(float(members))
    given = wp.float64(0.0)
    for index in range(room):
        spot = _spot(index, lows, lengths, filled)
        if owner[first + low + spot[0], low + spot[1], low + spot[2]] == cell:
            share = wp.float64(1.0)
            if kind == 1:
                share = field[spot[0], spot[1], spot[2]]
            exact = share / weight * total
            whole = wp.max(wp.floor(exact), wp.float64(1.0))
            field[spot[0], spot[1], spot[2]] = exact - wp.floor(exact)
            quanta[first + low + spot[0], low + spot[1], low + spot[2]] = kind * int(whole)
            given = given + whole
    rest = int(wp.rint(total - given))
    if rest <= 0:
        return
    bar = wp.float64(2.0)
    ties = members
    if rest < members:
        below = wp.float64(-1.0)
        bar = wp.float64(1.0)
        for round_index in range(2200):
            middle = wp.float64(0.5) * (below + bar)
            if middle <= below or middle >= bar:
                break
            above = int(0)
            for index in range(room):
                spot = _spot(index, lows, lengths, filled)
                if owner[first + low + spot[0], low + spot[1], low + spot[2]] == cell:
                    if field[spot[0], spot[1], spot[2]] > middle:
                        above += 1
            if above >= rest:
                below = middle
            else:
                bar = middle
        ties = rest
        for index in range(room):
            spot = _spot(index, lows, lengths, filled)
            if owner[first + low + spot[0], low + spot[1], low + spot[2]] == cell:
                if field[spot[0], spot[1], spot[2]] > bar:
                    ties -= 1
    for index in range(room):
        spot = _spot(index, lows, lengths, filled)
        if owner[first + low + spot[0], low + spot[1], low + spot[2]] == cell:
            left_over = field[spot[0], spot[1], spot[2]]
            more = int(0)
            if rest >= members or left_over > bar:
                more = 1
            elif left_over == bar and ties > 0:
                more = 1
                ties -= 1
            if more == 1:
                quanta[first + low + spot[0], low + spot[1], low + spot[2]] = quanta[
                    first + low + spot[0], low + spot[1], low + spot[2]] + kind
