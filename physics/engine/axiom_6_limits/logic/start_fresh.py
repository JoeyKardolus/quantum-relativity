"""Start fresh: every joint remembers the quanta it was made with.

Contract: `start_fresh(state)` writes `fresh_quanta[joint] = cell_quanta[joint]` for every
living joint and 0 for every other id. It runs once, at the start, after `count_cells`.
A joint made later by a tear gets its `fresh_quanta` from the tear. Nothing ever changes
it afterwards: a joint only gains energy (axiom 6), so its fresh quanta are the least it
will ever hold and its fresh length the fattest it will ever be.
Reads `kind`, `cell_quanta`. Writes `fresh_quanta` only.
"""
from __future__ import annotations

import warp as wp

from physics.engine.axiom_3_structure.definitions.kinds import JOINT

wp.set_module_options({"fuse_fp": False})


@wp.kernel
def _remember_fresh(kind: wp.array(dtype=wp.int32), cell_quanta: wp.array(dtype=wp.int64),
                    fresh_quanta: wp.array(dtype=wp.int64)):
    """One thread per cell id: a joint's fresh quanta are what it holds now."""
    cell = wp.tid()
    if kind[cell] == JOINT:
        fresh_quanta[cell] = cell_quanta[cell]
    else:
        fresh_quanta[cell] = wp.int64(0)


def start_fresh(state) -> None:
    """Write every joint's fresh quanta from what it holds at the start."""
    wp.launch(_remember_fresh, dim=state.count, inputs=[state.kind, state.cell_quanta, state.fresh_quanta],
              device=state.device)
