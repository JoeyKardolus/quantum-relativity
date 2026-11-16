"""Drain: a box whose cell no longer covers it pushes all its quanta out to the boxes of its own cell first.

Contract: `drain(state, phase_index)` runs rehome phase `phase_index` (0..26, the sites
of rehome_site.py) of a state with a cell lattice (none: returns at once). A site whose
target (box_targets.py) differs from its owner leaves its cell: its quanta flow out to
the boxes of its own cell among its 26 neighbours that hold quanta, each filled at most
to the cell's level, the larger of its packing (state.cell_quanta / state.cell_size,
rounded up: the packing the cell's moving region gives its energy) and the leaving
box's own quanta, so a drain never piles quanta past what the cell's size sets (a
receding border waits for the fluid to make room rather than heaping its quanta on a
few boxes). When the room is enough for all of them, the quanta are shared in
proportion to each box's room (whole numbers, the remainder to the first boxes with
room left), the site holds 0 and gets bit 4 in marks, provided the target cell, unless
it is the ring of empty space, has a face neighbour holding at least 2 quanta (the
giver hand_over.py takes half from); else every box with room is filled to the level,
the site keeps the rest and waits for a later step. A site also waits, untouched, when
leaving would break the cells' shape at the box level (where a cell is thinner than a
box the boxes cannot follow it; the simple-point test, one_group.py, read in the site's
3 x 3 x 3 block): a joint box that would stop being minus leaves only when the minus
boxes among its 18 face and edge neighbours stay one face-joined group (the web never
breaks, no joint box is sealed inside a volume); a volume box leaves only when the
boxes of its volume among its 26 neighbours stay one group; and a box joins an energy
volume only with no empty box, wall or box of another volume among its 26 neighbours
(every volume stays wrapped). Plus and minus never meet:
the quanta stay in their own cell and keep their sign (owner, 2026-11-16: a box changes
owner only after its energy has flowed out; no box ever holds two signs). The amount is
exact. hand_over.py, launched next for the same phase, gives the emptied site to its
new cell. Returns None.
"""
from __future__ import annotations

import warp as wp

from physics.engine.block_adjacency import block_adjacency
from physics.engine.cell_kind import cell_kind
from physics.engine.face_step import face_step
from physics.engine.inside import inside
from physics.engine.one_group import one_group
from physics.engine.rehome_site import rehome_dims, rehome_site
from physics.engine.state import State


@wp.func
def _room(level: int, held: int) -> int:
    """How many quanta a box holding `held` can take below `level`."""
    return wp.max(level - wp.abs(held), 0)


@wp.func
def _keeps_shape(quanta: wp.array3d(dtype=wp.int32), owner: wp.array3d(dtype=wp.int32), site: wp.vec3i,
                 home: int, side: int, own: int, goal: int, extent: wp.vec3i, cells: wp.vec3i,
                 adjacency: wp.array(dtype=wp.int32), face_adjacency: wp.array(dtype=wp.int32)) -> bool:
    """True when the site can leave `own` for `goal` without tearing a cell or the web, or unwrapping a volume."""
    own_kind = cell_kind(own, extent, cells)
    goal_kind = cell_kind(goal, extent, cells)
    same = int(0)
    minus = int(0)
    for index in range(27):
        if index != 13:
            offset = wp.vec3i(index // 9 - 1, index // 3 % 3 - 1, index % 3 - 1)
            there = site + offset
            if not inside(there, home, side):
                if goal_kind == 1:
                    return False
            else:
                held = quanta[there[0], there[1], there[2]]
                cell = owner[there[0], there[1], there[2]]
                if goal_kind == 1 and (held == 0 or (held > 0 and cell != goal)):
                    return False
                if cell == own and held != 0:
                    same = same | (1 << index)
                if held < 0 and wp.abs(offset[0]) + wp.abs(offset[1]) + wp.abs(offset[2]) <= 2:
                    minus = minus | (1 << index)
    if own_kind == 1 and not one_group(same, adjacency):
        return False
    if own_kind == -1 and goal_kind != -1 and (minus == 0 or not one_group(minus, face_adjacency)):
        return False
    return True


@wp.kernel
def _drain(quanta: wp.array3d(dtype=wp.int32), owner: wp.array3d(dtype=wp.int32),
           target: wp.array3d(dtype=wp.int32), marks: wp.array3d(dtype=wp.uint8),
           cell_quanta: wp.array2d(dtype=wp.int64), cell_size: wp.array2d(dtype=wp.float32), side: int,
           extent: wp.vec3i, counts: wp.array(dtype=wp.vec3i), phase_index: int, adjacency: wp.array(dtype=wp.int32),
           face_adjacency: wp.array(dtype=wp.int32)):
    home, ti, tj, tk = wp.tid()
    site, valid = rehome_site(home, ti, tj, tk, phase_index, side)
    if not valid:
        return
    own = owner[site[0], site[1], site[2]]
    goal = target[site[0], site[1], site[2]]
    if goal == own:
        return
    held = quanta[site[0], site[1], site[2]]
    amount = wp.abs(held)
    giver = int(0)
    for face in range(6):
        there = site + face_step(face)
        if inside(there, home, side):
            if owner[there[0], there[1], there[2]] == goal and wp.abs(quanta[there[0], there[1], there[2]]) >= 2:
                giver = 1
    cells = counts[home]
    if giver == 0 and cell_kind(goal, extent, cells) != 0:
        return
    if not _keeps_shape(quanta, owner, site, home, side, own, goal, extent, cells, adjacency, face_adjacency):
        return
    if amount > 0:
        size = cell_size[home, own]
        level = amount
        if size > 0.0:
            level = wp.max(level, int(wp.ceil(float(cell_quanta[home, own]) / size)))
        room = int(0)
        for index in range(27):
            there = site + wp.vec3i(index // 9 - 1, index // 3 % 3 - 1, index % 3 - 1)
            if index != 13 and inside(there, home, side):
                other = quanta[there[0], there[1], there[2]]
                if owner[there[0], there[1], there[2]] == own and other != 0:
                    room += _room(level, other)
        if room == 0:
            return
        given = int(0)
        for index in range(27):
            there = site + wp.vec3i(index // 9 - 1, index // 3 % 3 - 1, index % 3 - 1)
            if index != 13 and inside(there, home, side):
                other = quanta[there[0], there[1], there[2]]
                if owner[there[0], there[1], there[2]] == own and other != 0:
                    share = _room(level, other)
                    if room > amount:
                        share = int(wp.floor(float(share) * float(amount) / float(room)))
                    given += share
                    if held < 0:
                        share = -share
                    quanta[there[0], there[1], there[2]] = other + share
        if room > amount:
            for index in range(27):
                there = site + wp.vec3i(index // 9 - 1, index // 3 % 3 - 1, index % 3 - 1)
                if index != 13 and inside(there, home, side) and given < amount:
                    other = quanta[there[0], there[1], there[2]]
                    if owner[there[0], there[1], there[2]] == own and other != 0:
                        extra = wp.min(_room(level, other), amount - given)
                        given += extra
                        if held < 0:
                            extra = -extra
                        quanta[there[0], there[1], there[2]] = other + extra
        left = amount - given
        if held < 0:
            left = -left
        quanta[site[0], site[1], site[2]] = left
        if left != 0:
            return
    marks[site[0], site[1], site[2]] = marks[site[0], site[1], site[2]] | wp.uint8(4)


def drain(state: State, phase_index: int) -> None:
    """Empty the leaving boxes of rehome phase `phase_index` into their own cell's face neighbours."""
    if not state.has_lattice:
        return
    wp.launch(_drain, dim=rehome_dims(state, phase_index), device=state.quanta.device,
              inputs=[state.quanta, state.owner, state.target, state.marks, state.cell_quanta, state.cell_size,
                      state.side, wp.vec3i(*state.extent), state.counts, int(phase_index), block_adjacency(state.quanta.device),
                      block_adjacency(state.quanta.device, faces_only=True)])
