"""Mortar shift: a growing energy volume pushes a one-box-thick joint wall on into its neighbour, the wall keeping its thickness.

Contract: `mortar_shift(state, settings)` lets room pass from one energy volume to the
next through the joint they share where that joint is one box thick (a volume cannot
take such a joint box, it would touch its neighbour; exchange.py). Each step every
box of a joint J between volumes A and B is a site once (sites 5 apart, 125 phases, the
grid of sites moved by a fresh offset each step, so the 3 x 3 x 3 blocks round two
sites and the boxes they drain into never overlap). A site box j of J that touches A
across a face, has plus boxes of B among its 26 neighbours (the set S, n boxes) and no
other volume's, no empty space or wall, and every joint box on its faces a joint of A,
moves when A's deficit (state.size minus boxes) exceeds B's by at least n + 1, with
chance the slowest clock rate of A, B and J (on local clocks): j joins A and the boxes of S join J, so A grows by one box, B gives n,
J holds n - 1 more boxes than before, and its size (state.size) does not change: the
wall was moved, not grown, so J gives the extra boxes back to whichever volume wants
them (exchange.py). Each box of S
empties its plus quanta into the boxes of B on its faces that stay B's (whole quanta,
evenly) and takes an even share of j's minus quanta (at least 1 each); j takes half the
quanta of A's fullest face-neighbouring box with that box's phase and clock. A shift
waits when B would keep fewer than 2 boxes, a box of S has no box of B left on its
faces, B's boxes round a box of S would fall into two groups, the minus boxes round j
would not stay one face-joined group, or j holds fewer minus quanta than S has boxes.
No quantum changes sign, the amount is exact, ABAB holds. Every moved box is marked (2)
and counted in tallies[u, 2]. Returns the number of shifts.
"""
from __future__ import annotations

import numpy as np
import warp as wp

from physics.engine.block_adjacency import block_adjacency
from physics.engine.face_step import face_step
from physics.engine.one_group import one_group
from physics.engine.reach import reach
from physics.engine.settings import Settings
from physics.engine.state import State

_STRIDE = 5
_ORDER_SEED = 53


@wp.func
def _offset(index: int) -> wp.vec3i:
    """Block place `index` (0..26) as an offset from the centre."""
    return wp.vec3i(index // 9 - 1, index // 3 % 3 - 1, index % 3 - 1)


@wp.func
def _in_block(place: wp.vec3i, centre: wp.vec3i, home: int, side: int, wrap: int) -> bool:
    """True when `place` lies within one box of `centre` on every axis (round the edges when wrapping)."""
    for axis in range(3):
        apart = place[axis] - centre[axis]
        if axis == 0:
            apart = (place[0] - home * side) - (centre[0] - home * side)
        if wrap == 1:
            apart = ((apart % side) + side + side // 2) % side - side // 2
        if wp.abs(apart) > 1:
            return False
    return True


@wp.func
def _scan(quanta: wp.array3d(dtype=wp.int32), owner: wp.array3d(dtype=wp.int32), between: wp.array2d(dtype=wp.vec2i),
          site: wp.vec3i, home: int, side: int, wrap: int, grower: int, giver_cell: int) -> int:
    """How many of the giver's plus boxes lie round the site, or -1 when the site cannot join the grower."""
    count = int(0)
    touches = int(0)
    for index in range(27):
        if index != 13:
            offset = _offset(index)
            there, valid = reach(site, offset, home, side, wrap)
            if not valid:
                return -1
            held = quanta[there[0], there[1], there[2]]
            cell = owner[there[0], there[1], there[2]]
            if held == 0:
                return -1
            distance = wp.abs(offset[0]) + wp.abs(offset[1]) + wp.abs(offset[2])
            if held > 0:
                if cell == giver_cell:
                    count += 1
                elif cell != grower:
                    return -1
                elif distance == 1:
                    touches = 1
            elif distance == 1:
                sides = between[home, cell]
                if sides[0] != grower and sides[1] != grower:
                    return -1
    if touches == 0:
        return -1
    return count


@wp.kernel
def _shift(quanta: wp.array3d(dtype=wp.int32), phase: wp.array3d(dtype=wp.float32),
           clock: wp.array3d(dtype=wp.float32), owner: wp.array3d(dtype=wp.int32), marks: wp.array3d(dtype=wp.uint8),
           kinds: wp.array2d(dtype=wp.int32), between: wp.array2d(dtype=wp.vec2i), sizes: wp.array2d(dtype=wp.float32),
           rates: wp.array2d(dtype=wp.float32), boxes: wp.array2d(dtype=wp.int32),
           tallies: wp.array2d(dtype=wp.int64), side: int, wrap: int, span: int, offset: wp.vec3i, chance: float,
           seed: int, keys: wp.array(dtype=wp.int32), adjacency: wp.array(dtype=wp.int32),
           face_adjacency: wp.array(dtype=wp.int32)):
    home, ti, tj, tk = wp.tid()
    steps = wp.vec3i(ti, tj, tk)
    local = wp.vec3i(0, 0, 0)
    for axis in range(3):
        if _STRIDE * steps[axis] > span:
            return
        local[axis] = (offset[axis] + _STRIDE * steps[axis]) % side
    site = wp.vec3i(home * side + local[0], local[1], local[2])
    if (marks[site[0], site[1], site[2]] & wp.uint8(2)) != wp.uint8(0):
        return
    joint = owner[site[0], site[1], site[2]]
    if joint < 0 or kinds[home, joint] != -1:
        return
    sides = between[home, joint]
    if sides[0] < 0 or sides[1] < 0:
        return
    grower = int(-1)
    shrinker = int(-1)
    count = int(0)
    for pick in range(2):
        first = sides[pick]
        second = sides[1 - pick]
        found = _scan(quanta, owner, between, site, home, side, wrap, first, second)
        if found > 0:
            lead = (sizes[home, first] - float(boxes[home, first])) - (sizes[home, second] - float(boxes[home, second]))
            if lead >= float(found + 1):
                grower = first
                shrinker = second
                count = found
    if grower < 0:
        return
    rate = wp.min(rates[home, joint], wp.min(rates[home, grower], rates[home, shrinker]))
    index = (local[0] * side + local[1]) * side + local[2]
    rng = wp.rand_init(seed ^ keys[home], index)
    if wp.randf(rng) >= chance * rate:
        return
    held = -quanta[site[0], site[1], site[2]]
    if held < count:
        return
    giver = wp.vec3i(0, 0, 0)
    given = int(1)
    for face in range(6):
        there, valid = reach(site, face_step(face), home, side, wrap)
        if valid and owner[there[0], there[1], there[2]] == grower and quanta[there[0], there[1], there[2]] > given:
            if (marks[there[0], there[1], there[2]] & wp.uint8(2)) == wp.uint8(0):
                given = quanta[there[0], there[1], there[2]]
                giver = there
    if given < 2:
        return
    minus = int(0)
    for index_place in range(27):
        if index_place != 13:
            offset_place = _offset(index_place)
            there, valid = reach(site, offset_place, home, side, wrap)
            distance = wp.abs(offset_place[0]) + wp.abs(offset_place[1]) + wp.abs(offset_place[2])
            if distance <= 2:
                cell = owner[there[0], there[1], there[2]]
                if quanta[there[0], there[1], there[2]] < 0 or cell == shrinker:
                    minus = minus | (1 << index_place)
    if minus == 0 or not one_group(minus, face_adjacency):
        return
    for index_place in range(27):
        if index_place != 13:
            near, valid = reach(site, _offset(index_place), home, side, wrap)
            if owner[near[0], near[1], near[2]] == shrinker:
                stays = int(0)
                receivers = int(0)
                for around in range(27):
                    if around != 13:
                        spot, inside = reach(near, _offset(around), home, side, wrap)
                        if inside and owner[spot[0], spot[1], spot[2]] == shrinker and not _in_block(
                                spot, site, home, side, wrap):
                            stays = stays | (1 << around)
                            step_size = wp.abs(_offset(around)[0]) + wp.abs(_offset(around)[1]) + wp.abs(
                                _offset(around)[2])
                            if step_size == 1:
                                receivers += 1
                if receivers == 0 or not one_group(stays, adjacency):
                    return
    taken = wp.atomic_add(boxes, home, grower, 1)
    left = wp.atomic_add(boxes, home, shrinker, -count)
    lead = (sizes[home, grower] - float(taken)) - (sizes[home, shrinker] - float(left))
    if left - count < 2 or lead < float(count + 1):
        wp.atomic_add(boxes, home, grower, -1)
        wp.atomic_add(boxes, home, shrinker, count)
        return
    wp.atomic_add(boxes, home, joint, count - 1)
    each = held // count
    rest = held - each * count
    for index_place in range(27):
        if index_place != 13:
            near, valid = reach(site, _offset(index_place), home, side, wrap)
            if owner[near[0], near[1], near[2]] == shrinker:
                amount = quanta[near[0], near[1], near[2]]
                receivers = int(0)
                for face in range(6):
                    spot, inside = reach(near, face_step(face), home, side, wrap)
                    if inside and owner[spot[0], spot[1], spot[2]] == shrinker and not _in_block(spot, site, home,
                                                                                               side, wrap):
                        receivers += 1
                share = amount // receivers
                extra = amount - share * receivers
                for face in range(6):
                    spot, inside = reach(near, face_step(face), home, side, wrap)
                    if inside and owner[spot[0], spot[1], spot[2]] == shrinker and not _in_block(spot, site, home,
                                                                                               side, wrap):
                        bonus = int(0)
                        if extra > 0:
                            bonus = 1
                            extra -= 1
                        quanta[spot[0], spot[1], spot[2]] = quanta[spot[0], spot[1], spot[2]] + share + bonus
                portion = each
                if rest > 0:
                    portion += 1
                    rest -= 1
                quanta[near[0], near[1], near[2]] = -portion
                owner[near[0], near[1], near[2]] = joint
                phase[near[0], near[1], near[2]] = phase[site[0], site[1], site[2]]
                clock[near[0], near[1], near[2]] = clock[site[0], site[1], site[2]]
                marks[near[0], near[1], near[2]] = marks[near[0], near[1], near[2]] | wp.uint8(2)
    half = given // 2
    quanta[giver[0], giver[1], giver[2]] = given - half
    quanta[site[0], site[1], site[2]] = half
    phase[site[0], site[1], site[2]] = phase[giver[0], giver[1], giver[2]]
    clock[site[0], site[1], site[2]] = clock[giver[0], giver[1], giver[2]]
    owner[site[0], site[1], site[2]] = grower
    marks[site[0], site[1], site[2]] = marks[site[0], site[1], site[2]] | wp.uint8(2)
    wp.atomic_add(tallies, home, 2, wp.int64(count + 1))


def mortar_shift(state: State, settings: Settings) -> int:
    """Push one-box-thick joint walls from growing volumes into their neighbours; returns the shifts made."""
    device = state.quanta.device
    boxes = wp.array(state.cell_boxes.astype(np.int32), dtype=wp.int32, device=device)
    before = int(state.tallies.numpy()[:, 2].sum())
    span = state.side - _STRIDE if state.wrap else state.side - 1
    rng = np.random.default_rng((_ORDER_SEED, state.step))
    shift = rng.integers(0, state.side, size=3) if state.wrap else np.zeros(3, dtype=np.int64)
    order = rng.permutation(_STRIDE ** 3)
    dims = (state.universes,) + (span // _STRIDE + 1,) * 3
    adjacency = block_adjacency(device)
    faces = block_adjacency(device, faces_only=True)
    for phase_index in order:
        place = np.array([phase_index // 25, phase_index // 5 % 5, phase_index % 5]) + shift
        seed = (state.step * 7919 + int(phase_index) * 197 + 3) & 0x7FFFFFFF
        wp.launch(_shift, dim=dims, device=device,
                  inputs=[state.quanta, state.phase, state.clock, state.owner, state.marks, state.cells_d["kind"],
                          state.cells_d["between"], state.cells_d["size"], state.cells_d["rate"], boxes,
                          state.tallies, state.side, int(state.wrap), int(span), wp.vec3i(*[int(v) for v in place]),
                          1.0, int(seed), state.keys, adjacency, faces])
    state.cell_boxes = boxes.numpy().astype(np.int64)
    return int(state.tallies.numpy()[:, 2].sum()) - before
