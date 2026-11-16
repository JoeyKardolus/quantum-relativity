"""Collapse: a joint box squeezed past its limit gives way, its minus quanta becoming plus quanta, amount kept.

Contract: `collapse(state, settings)` flips every joint box (minus) whose squeeze
(squeeze.py) is at least collapse_squeeze, that has a plus box among its 26 neighbours
(it lies between energy volumes) and no empty box or wall among them (a joint touching
empty space does not collapse, and an energy volume never touches empty space): its
quanta n < 0 become -n > 0, the same amount, the opposite sign (bible rule 4: joint
collapse is the one place where pushing energy becomes pulling energy; the packed volume
it joins is where a black hole begins). The energy volumes on either side are joined
through it: they fall into each other. Two passes, so every box decides on the same
signs (marks first, flips after); the flips can run away over later steps as the joined
volume pulls harder. tallies[u, 1] counts universe u's flipped boxes and tallies[u, 3]
the quanta they turned plus; a flipped box's marks get bit 1 (bit 2, a claim's, is
kept). Returns None.
"""
from __future__ import annotations

import warp as wp

from physics.engine.inside import inside
from physics.engine.settings import Settings
from physics.engine.squeeze import squeeze
from physics.engine.state import State


@wp.kernel
def _mark(quanta: wp.array3d(dtype=wp.int32), marks: wp.array3d(dtype=wp.uint8), side: int, joint_rest: float,
          limit: float):
    ti, tj, tk = wp.tid()
    marks[ti, tj, tk] = marks[ti, tj, tk] & wp.uint8(2)
    held = quanta[ti, tj, tk]
    if held >= 0 or squeeze(held, joint_rest) < limit:
        return
    site = wp.vec3i(ti, tj, tk)
    has_plus = int(0)
    for index in range(27):
        if index != 13:
            there = site + wp.vec3i(index // 9 - 1, index // 3 % 3 - 1, index % 3 - 1)
            if not inside(there, ti // side, side):
                return
            value = quanta[there[0], there[1], there[2]]
            if value == 0:
                return
            if value > 0:
                has_plus = 1
    if has_plus == 1:
        marks[ti, tj, tk] = marks[ti, tj, tk] | wp.uint8(1)


@wp.kernel
def _flip(quanta: wp.array3d(dtype=wp.int32), marks: wp.array3d(dtype=wp.uint8), side: int,
          tallies: wp.array2d(dtype=wp.int64)):
    ti, tj, tk = wp.tid()
    if (marks[ti, tj, tk] & wp.uint8(1)) == wp.uint8(1):
        amount = -quanta[ti, tj, tk]
        quanta[ti, tj, tk] = amount
        wp.atomic_add(tallies, ti // side, 1, wp.int64(1))
        wp.atomic_add(tallies, ti // side, 3, wp.int64(amount))


def collapse(state: State, settings: Settings) -> None:
    """Flip every joint box squeezed past the collapse limit between energy volumes: minus quanta become plus."""
    shape = state.shape
    device = state.quanta.device
    wp.launch(_mark, dim=shape, device=device,
              inputs=[state.quanta, state.marks, state.side, settings.joint_rest, settings.collapse_squeeze])
    wp.launch(_flip, dim=shape, device=device, inputs=[state.quanta, state.marks, state.side, state.tallies])
