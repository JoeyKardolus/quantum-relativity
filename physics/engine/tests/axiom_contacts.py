"""Axiom contacts (test helper): count touching energy volumes and open volume boxes in the boxes, on the GPU.

Contract: `axiom_contacts(state)` returns (touching, open_boxes) summed over the batch:
touching is the number of box pairs, two plus boxes of two different cells that are
26-neighbours (each pair once); open_boxes the plus boxes with an empty box or a wall
among their 26 neighbours (round the edges when the universe wraps: then there is no
wall). Both are 0 while every energy volume is wrapped by joints on all 26 sides (bible
"The foundation" 1). The state is not changed.
"""
from __future__ import annotations

import warp as wp

from physics.engine import State, reach


@wp.kernel
def _count(quanta: wp.array3d(dtype=wp.int32), owner: wp.array3d(dtype=wp.int32), side: int, wrap: int,
           sums: wp.array(dtype=wp.int64)):
    ti, tj, tk = wp.tid()
    if quanta[ti, tj, tk] <= 0:
        return
    here = wp.vec3i(ti, tj, tk)
    opened = int(0)
    for index in range(27):
        if index != 13:
            there, valid = reach(here, wp.vec3i(index // 9 - 1, index // 3 % 3 - 1, index % 3 - 1), ti // side, side,
                                 wrap)
            if not valid:
                opened = 1
            else:
                held = quanta[there[0], there[1], there[2]]
                if held == 0:
                    opened = 1
                elif held > 0 and owner[there[0], there[1], there[2]] != owner[ti, tj, tk]:
                    wp.atomic_add(sums, 0, wp.int64(1))
    if opened == 1:
        wp.atomic_add(sums, 1, wp.int64(1))


def axiom_contacts(state: State) -> tuple:
    """(touching box pairs of two different volumes, plus boxes beside empty space or a wall)."""
    sums = wp.zeros(2, dtype=wp.int64, device=state.quanta.device)
    wp.launch(_count, dim=state.shape, device=state.quanta.device, inputs=[state.quanta, state.owner, state.side, int(state.wrap), sums])
    touching, open_boxes = (int(value) for value in sums.numpy())
    return touching // 2, open_boxes
