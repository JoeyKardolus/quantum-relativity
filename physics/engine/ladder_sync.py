"""Ladder sync: the engine's once-a-step pass over the detail ladder kept beside its boxes.

Contract: `ladder_sync(state)` does nothing and returns None when state.ladder is None
(attach_ladder.py). Otherwise it cuts the boxes' quanta and phase into blocks of 8**3
(in the store's block order, on the device), hands them to physics.ladder.write_blocks
(bit-identical blocks are skipped; changed ones are stored FULL exactly as the engine
holds them) and runs physics.ladder.switch_levels (empty blocks drop to EMPTY, blocks
with a whirl or flow stay FULL, quiet ones go COARSE or REUSED). Returns the switch's
counts plus "written", the blocks that changed. The engine's own boxes stay the truth
that step() advances: the ladder holds the same universe at the detail each block's
quietness allows, with plus and minus totals exact (physics.ladder.total_quanta), and
says what the universe costs counted the ladder way (physics.ladder.memory_per_box). A
ladder block's quietness is measured with a one-box border that, on the first axis, may
reach into the next universe of a batch; it decides only a block's level, never a value.
"""
from __future__ import annotations

import numpy as np
import warp as wp

from physics.engine.state import State
from physics.ladder import switch_levels, write_blocks
from physics.ladder.settings import BLOCK


@wp.kernel
def _cut(quanta: wp.array3d(dtype=wp.int32), phase: wp.array3d(dtype=wp.float32), blocks_y: int, blocks_z: int,
         out_quanta: wp.array4d(dtype=wp.int32), out_phase: wp.array4d(dtype=wp.float32)):
    row, bi, bj, bk = wp.tid()
    across = (row // (blocks_y * blocks_z)) * 8 + bi
    up = ((row // blocks_z) % blocks_y) * 8 + bj
    deep = (row % blocks_z) * 8 + bk
    out_quanta[row, bi, bj, bk] = quanta[across, up, deep]
    out_phase[row, bi, bj, bk] = phase[across, up, deep]


def ladder_sync(state: State) -> dict | None:
    """Write the changed blocks into the attached ladder and switch every block's level."""
    store = state.ladder
    if store is None:
        return None
    count = int(np.prod(store.blocks))
    device = state.quanta.device
    quanta = wp.empty((count, BLOCK, BLOCK, BLOCK), dtype=wp.int32, device=device)
    phase = wp.empty((count, BLOCK, BLOCK, BLOCK), dtype=wp.float32, device=device)
    wp.launch(_cut, dim=(count, BLOCK, BLOCK, BLOCK), device=device,
              inputs=[state.quanta, state.phase, store.blocks[1], store.blocks[2]], outputs=[quanta, phase])
    coords = np.argwhere(np.ones(store.blocks, dtype=bool))
    written = write_blocks(store, coords, quanta, phase)
    return switch_levels(store) | {"written": int(written)}
