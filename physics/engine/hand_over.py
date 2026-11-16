"""Hand over: an emptied box joins the cell that covers it, taking half the quanta of that cell's fullest face neighbour.

Contract: `hand_over(state, phase_index)` runs rehome phase `phase_index` (0..26) right
after drain(state, phase_index): every site drain emptied (marks bit 4) holds 0 quanta;
its owner becomes its target (box_targets.py), and unless that is the ring of empty
space it takes half (rounded down in amount, sign kept) of the quanta of its face
neighbour of the new cell holding the most (at least 2; the first such face on a tie),
with that giver's phase and clock reading. Bit 4 is cleared, bit 2 set, and
tallies[u, 2] counts the owner changes. A box changes owner only here, only while
empty, and its new quanta come from its new cell, so no box ever holds two signs and
no quantum changes sign; the amount is exact. A box handed to the ring is empty space.
Returns None.
"""
from __future__ import annotations

import warp as wp

from physics.engine.face_step import face_step
from physics.engine.inside import inside
from physics.engine.rehome_site import rehome_dims, rehome_site
from physics.engine.state import State


@wp.kernel
def _hand_over(quanta: wp.array3d(dtype=wp.int32), phase: wp.array3d(dtype=wp.float32),
               clock: wp.array3d(dtype=wp.float32), owner: wp.array3d(dtype=wp.int32),
               target: wp.array3d(dtype=wp.int32), marks: wp.array3d(dtype=wp.uint8), side: int,
               phase_index: int, tallies: wp.array2d(dtype=wp.int64)):
    home, ti, tj, tk = wp.tid()
    site, valid = rehome_site(home, ti, tj, tk, phase_index, side)
    if not valid:
        return
    flags = marks[site[0], site[1], site[2]]
    if (flags & wp.uint8(4)) == wp.uint8(0):
        return
    goal = target[site[0], site[1], site[2]]
    best = int(1)
    best_face = int(-1)
    for face in range(6):
        there = site + face_step(face)
        if inside(there, home, side):
            other = wp.abs(quanta[there[0], there[1], there[2]])
            if owner[there[0], there[1], there[2]] == goal and other > best:
                best = other
                best_face = face
    owner[site[0], site[1], site[2]] = goal
    phase[site[0], site[1], site[2]] = 0.0
    if best_face >= 0:
        giver = site + face_step(best_face)
        given = quanta[giver[0], giver[1], giver[2]]
        half = wp.abs(given) // 2
        if given < 0:
            half = -half
        quanta[giver[0], giver[1], giver[2]] = given - half
        quanta[site[0], site[1], site[2]] = half
        phase[site[0], site[1], site[2]] = phase[giver[0], giver[1], giver[2]]
        clock[site[0], site[1], site[2]] = clock[giver[0], giver[1], giver[2]]
    marks[site[0], site[1], site[2]] = (flags & wp.uint8(251)) | wp.uint8(2)
    wp.atomic_add(tallies, home, 2, wp.int64(1))


def hand_over(state: State, phase_index: int) -> None:
    """Give the emptied boxes of rehome phase `phase_index` to the cells that cover them."""
    if not state.has_lattice:
        return
    wp.launch(_hand_over, dim=rehome_dims(state, phase_index), device=state.quanta.device,
              inputs=[state.quanta, state.phase, state.clock, state.owner, state.target, state.marks, state.side,
                      int(phase_index), state.tallies])
