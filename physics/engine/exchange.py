"""Exchange: the boxes follow the cells' sizes, one wall box at a time, each moving only once its quanta have flowed out.

Contract: `exchange(state, settings)` runs every exchange phase once
(state.side split into sites `stride` apart on each axis, stride**3 phases, in an order
shuffled afresh each step; sites of one phase share no box of their 3 x 3 x 3 blocks).
Each cell has a deficit, its size (state.size: an energy volume's is the size its energy
wants, a joint's its swinging size) minus its boxes. A site box of cell X moves to the
face-neighbouring cell Y with the largest deficit when Y's deficit exceeds X's by at
least 2 (so the move never turns the order round) (where a joint is one box thick a
volume cannot take its boxes; mortar_shift.py moves such a wall); with chance the slower of
the two cells' clock rates (the boxes follow on local clocks, at most one box a step),
when all of these hold:
  - X and Y are not two energy volumes (they never touch), and an energy volume trades
    boxes only with its own joints (state.between): the graph does not change
  - the box has not changed cell this step, and Y has a face-neighbouring box that has
    not either and holds at least 2 quanta (the giver), and X keeps at least 2 boxes
  - joining an energy volume: all 26 neighbours are space and none is another
    volume's (every volume stays wrapped on all 26 sides; it may come to touch a joint
    of two other volumes, which carries nothing between it and them, cross.py)
  - joining a joint: every plus box on its faces belongs to one of Y's two volumes
  - leaving: an energy volume's boxes round it stay one group (a joint may come apart
    in pieces, the web may not), and a joint box turning into a volume box leaves the minus boxes among its 26 neighbours one face-joined group, so they stay
    joined through the block (the web never breaks)
The box then empties into X's boxes among its 26 neighbours (whole quanta, shared out
evenly, sign kept), joins Y and takes half the quanta of the fullest giver with the giver's phase and clock:
no box ever holds two signs and no quantum changes sign; the amount is exact. An empty
box beside a joint is taken by that joint the same way, with its clock rate as chance and no
deficit asked (a joint only pushes and empty space does nothing; not in the
correction): this is how space grows where joints meet empty space. Box counts are
updated with atomics as boxes move, and a move that would break the deficit order is
put back. Every move marks its box (2) and adds to tallies[u, 2]. A joint's size
follows the boxes it took from energy volumes or empty space and the boxes it gave to
energy volumes out of its own size (not out of boxes it held beyond its size)
(state.size), so only the joints' own swing moves boxes between joints. state.cell_boxes is updated.
Returns the number of boxes moved.
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

_ORDER_SEED = 27


@wp.func
def _holds(sides: wp.vec2i, cell: int) -> bool:
    """True when `cell` is one of a joint's two volumes."""
    return cell >= 0 and (sides[0] == cell or sides[1] == cell)


@wp.func
def _fits(quanta: wp.array3d(dtype=wp.int32), owner: wp.array3d(dtype=wp.int32), kinds: wp.array2d(dtype=wp.int32),
          between: wp.array2d(dtype=wp.vec2i), site: wp.vec3i, home: int, side: int, wrap: int, own: int, goal: int,
          adjacency: wp.array(dtype=wp.int32), face_adjacency: wp.array(dtype=wp.int32)) -> bool:
    """True when the site may leave `own` (-1: empty) for `goal` without breaking a cell, the web or the wrapping."""
    goal_kind = kinds[home, goal]
    own_kind = int(0)
    if own >= 0:
        own_kind = kinds[home, own]
    same = int(0)
    minus = int(0)
    for index in range(27):
        if index != 13:
            offset = wp.vec3i(index // 9 - 1, index // 3 % 3 - 1, index % 3 - 1)
            distance = wp.abs(offset[0]) + wp.abs(offset[1]) + wp.abs(offset[2])
            there, valid = reach(site, offset, home, side, wrap)
            if not valid:
                if goal_kind == 1:
                    return False
            else:
                held = quanta[there[0], there[1], there[2]]
                cell = owner[there[0], there[1], there[2]]
                if goal_kind == 1 and (held == 0 or (held > 0 and cell != goal)):
                    return False
                if goal_kind == -1 and held > 0 and distance == 1 and not _holds(between[home, goal], cell):
                    return False
                if own >= 0 and cell == own:
                    same = same | (1 << index)
                if held < 0:
                    minus = minus | (1 << index)
    if own_kind == 1 and not one_group(same, adjacency):
        return False
    if own_kind == -1 and goal_kind == 1 and (minus == 0 or not one_group(minus, face_adjacency)):
        return False
    return True


@wp.kernel
def _exchange(quanta: wp.array3d(dtype=wp.int32), phase: wp.array3d(dtype=wp.float32),
              clock: wp.array3d(dtype=wp.float32), owner: wp.array3d(dtype=wp.int32),
              marks: wp.array3d(dtype=wp.uint8), kinds: wp.array2d(dtype=wp.int32),
              between: wp.array2d(dtype=wp.vec2i), sizes: wp.array2d(dtype=wp.float32),
              rates: wp.array2d(dtype=wp.float32), boxes: wp.array2d(dtype=wp.int32), sync: wp.array2d(dtype=wp.int32),
              tallies: wp.array2d(dtype=wp.int64), side: int, wrap: int, stride: int, offset: wp.vec3i, chance: float,
              seed: int, keys: wp.array(dtype=wp.int32), adjacency: wp.array(dtype=wp.int32),
              face_adjacency: wp.array(dtype=wp.int32)):
    home, ti, tj, tk = wp.tid()
    local = wp.vec3i(offset[0] + stride * ti, offset[1] + stride * tj, offset[2] + stride * tk)
    if local[0] >= side or local[1] >= side or local[2] >= side:
        return
    site = wp.vec3i(home * side + local[0], local[1], local[2])
    if (marks[site[0], site[1], site[2]] & wp.uint8(2)) != wp.uint8(0):
        return
    own = owner[site[0], site[1], site[2]]
    own_kind = int(0)
    own_room = float(0.0)
    if own >= 0:
        own_kind = kinds[home, own]
        own_room = sizes[home, own] - float(boxes[home, own])
    goal = int(-1)
    best = float(-1.0e30)
    for face in range(6):
        there, valid = reach(site, face_step(face), home, side, wrap)
        if valid:
            cell = owner[there[0], there[1], there[2]]
            if cell >= 0 and cell != own:
                cell_kind = kinds[home, cell]
                allowed = True
                room = sizes[home, cell] - float(boxes[home, cell])
                if own < 0:
                    allowed = cell_kind == -1
                    room = float(1.0e20)
                elif own_kind == 1 and cell_kind == 1:
                    allowed = False
                elif own_kind == 1 and not _holds(between[home, cell], own):
                    allowed = False
                elif cell_kind == 1 and own_kind == -1 and not _holds(between[home, own], cell):
                    allowed = False
                elif room - own_room < 2.0:
                    allowed = False
                if allowed and room > best:
                    best = room
                    goal = cell
    if goal < 0:
        return
    rate = rates[home, goal]
    if own >= 0:
        rate = wp.min(rate, rates[home, own])
    index = (local[0] * side + local[1]) * side + local[2]
    rng = wp.rand_init(seed ^ keys[home], index)
    if wp.randf(rng) >= chance * rate:
        return
    giver = wp.vec3i(0, 0, 0)
    given = int(1)
    receivers = int(0)
    for index_place in range(27):
        if index_place != 13:
            there, valid = reach(site, wp.vec3i(index_place // 9 - 1, index_place // 3 % 3 - 1, index_place % 3 - 1),
                                 home, side, wrap)
            if valid:
                cell = owner[there[0], there[1], there[2]]
                held = wp.abs(quanta[there[0], there[1], there[2]])
                if cell == goal and held > given and (marks[there[0], there[1], there[2]] & wp.uint8(2)) == wp.uint8(0):
                    given = held
                    giver = there
    for index_place in range(27):
        if index_place != 13:
            there, valid = reach(site, wp.vec3i(index_place // 9 - 1, index_place // 3 % 3 - 1, index_place % 3 - 1),
                                 home, side, wrap)
            if valid and own >= 0 and owner[there[0], there[1], there[2]] == own and quanta[there[0], there[1],
                                                                                         there[2]] != 0:
                receivers += 1
    if given < 2 or (own >= 0 and receivers == 0):
        return
    if not _fits(quanta, owner, kinds, between, site, home, side, wrap, own, goal, adjacency, face_adjacency):
        return
    if own >= 0:
        left = wp.atomic_add(boxes, home, own, -1)
        taken = wp.atomic_add(boxes, home, goal, 1)
        order = (sizes[home, goal] - float(taken)) - (sizes[home, own] - float(left))
        if left < 2 or order < 2.0:
            wp.atomic_add(boxes, home, own, 1)
            wp.atomic_add(boxes, home, goal, -1)
            return
    else:
        wp.atomic_add(boxes, home, goal, 1)
    held = quanta[site[0], site[1], site[2]]
    amount = wp.abs(held)
    if own >= 0:
        each = amount // receivers
        rest = amount - each * receivers
        for index_place in range(27):
            there, valid = reach(site, wp.vec3i(index_place // 9 - 1, index_place // 3 % 3 - 1, index_place % 3 - 1),
                                 home, side, wrap)
            if index_place != 13 and valid and owner[there[0], there[1], there[2]] == own:
                other = quanta[there[0], there[1], there[2]]
                if other != 0:
                    extra = each
                    if rest > 0:
                        extra += 1
                        rest -= 1
                    if held < 0:
                        extra = -extra
                    quanta[there[0], there[1], there[2]] = other + extra
    gift = quanta[giver[0], giver[1], giver[2]]
    half = wp.abs(gift) // 2
    if gift < 0:
        half = -half
    quanta[giver[0], giver[1], giver[2]] = gift - half
    quanta[site[0], site[1], site[2]] = half
    phase[site[0], site[1], site[2]] = phase[giver[0], giver[1], giver[2]]
    clock[site[0], site[1], site[2]] = clock[giver[0], giver[1], giver[2]]
    owner[site[0], site[1], site[2]] = goal
    marks[site[0], site[1], site[2]] = marks[site[0], site[1], site[2]] | wp.uint8(2)
    wp.atomic_add(tallies, home, 2, wp.int64(1))
    goal_kind = kinds[home, goal]
    if goal_kind == -1 and own_kind != -1:
        wp.atomic_add(sync, home, goal, 1)
    if own_kind == -1 and goal_kind != -1 and float(left - 1) < sizes[home, own] - 0.5:
        wp.atomic_add(sync, home, own, -1)


def _stride(state: State) -> int:
    """The spacing of one phase's sites: 3, or the smallest whole divisor of an even wrapping side from 3 up."""
    if not state.wrap:
        return 3
    return next(value for value in range(3, state.side + 1) if state.side % value == 0)


def exchange(state: State, settings: Settings) -> int:
    """Move wall boxes towards the cells whose size wants them, every phase once; returns the boxes moved."""
    device = state.quanta.device
    stride = _stride(state)
    boxes = wp.array(state.cell_boxes.astype(np.int32), dtype=wp.int32, device=device)
    sync = wp.zeros(state.kind.shape, dtype=wp.int32, device=device)
    before = int(state.tallies.numpy()[:, 2].sum())
    chance = 1.0
    order = np.random.default_rng((_ORDER_SEED, state.step)).permutation(stride ** 3)
    dims = (state.universes,) + (-(-state.side // stride),) * 3
    adjacency = block_adjacency(device)
    faces = block_adjacency(device, faces_only=True)
    for phase_index in order:
        offset = wp.vec3i(int(phase_index) // (stride * stride), int(phase_index) // stride % stride,
                          int(phase_index) % stride)
        seed = (state.step * 4099 + int(phase_index) * 131 + 7) & 0x7FFFFFFF
        wp.launch(_exchange, dim=dims, device=device,
                  inputs=[state.quanta, state.phase, state.clock, state.owner, state.marks, state.cells_d["kind"],
                          state.cells_d["between"], state.cells_d["size"], state.cells_d["rate"], boxes, sync,
                          state.tallies, state.side, int(state.wrap), stride, offset, float(chance),
                          int(seed), state.keys, adjacency, faces])
    state.cell_boxes = boxes.numpy().astype(np.int64)
    state.size = np.where(state.kind == -1, state.size + sync.numpy(), state.size)
    return int(state.tallies.numpy()[:, 2].sum()) - before
