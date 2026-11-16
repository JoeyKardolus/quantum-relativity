"""Measure quietness: how much flow, whirl and space each block holds now.

Contract: `measure_quietness(store, block_ids)` takes an (n, 3) array of block
coordinates and returns a dict of host arrays, one entry per block, measured on the block
and a halo of one box (so a whirl or flow at a block's face counts on both sides):
  filled  boxes with space (nonzero quanta) in the block; 0 means the block can be EMPTY
  flow    the largest wrapped phase step (radians) across a face from one of the block's
          boxes with space to a neighbour with space (the fluid's flow, hbar = mass = 1)
  whirls  unit squares of boxes with space, touching the block, round which the phase
          makes a whole turn (count of square-corner hits; > 0 means a whirl is there)
  busy    whirls > 0 or flow > settings.flow_tol: the block must be FULL
Measured in batches of `settings.chunk_blocks`.
"""
from __future__ import annotations

import numpy as np
import warp as wp

from physics.ladder._kernels import quiet_kernel
from physics.ladder.gather_blocks import gather_blocks
from physics.ladder.settings import BLOCK
from physics.ladder.store import LadderStore


def measure_quietness(store: LadderStore, block_ids) -> dict:
    """Per block: boxes with space, largest flow, whole-turn squares, and whether it must stay FULL."""
    coords = np.asarray(block_ids, dtype=np.int64).reshape(-1, 3)
    parts = {"filled": [], "flow": [], "whirls": []}
    chunk = max(1, int(store.settings.chunk_blocks))
    for start in range(0, coords.shape[0], chunk):
        piece = coords[start:start + chunk]
        quanta, phase = gather_blocks(store, piece, halo=1)
        count = piece.shape[0]
        flow = wp.zeros(count, dtype=wp.float32, device=store.device)
        whirls = wp.zeros(count, dtype=wp.int32, device=store.device)
        filled = wp.zeros(count, dtype=wp.int32, device=store.device)
        wp.launch(quiet_kernel, dim=(count, BLOCK, BLOCK, BLOCK), device=store.device,
                  inputs=[quanta, phase], outputs=[flow, whirls, filled])
        parts["flow"].append(flow.numpy())
        parts["whirls"].append(whirls.numpy())
        parts["filled"].append(filled.numpy())
    result = {name: (np.concatenate(values) if values else np.zeros(0)) for name, values in parts.items()}
    result["busy"] = (result["whirls"] > 0) | (result["flow"] > store.settings.flow_tol)
    return result
