"""New quanta batch: a seeded batch of universes, filled on the GPU so huge sizes need no host copy.

Contract: `new_quanta_batch(universe_count, side, seed=1, largest=8, device="cuda")`
returns a QuantaBatch (quanta_batch.py) with steps 0 whose every box holds a whole
number drawn uniformly from -largest..largest (zero included: some boxes are no space)
and a phase uniform in [-pi, pi), from a counter-based generator keyed on the seed and
the box's index, so the same arguments give the same batch. Universes differ from each
other (each has its own boxes' indices). Raises ValueError for universe_count < 1,
side < 2, largest < 1, or 2**31 boxes or more (the int32 box index), and MemoryError,
before allocating, when the two arrays (8 bytes a box) would not fit in the device's
free memory: a size search counts that size as too big.
"""
from __future__ import annotations

import warp as wp

from physics.bench.quanta_batch import QuantaBatch

BYTES_PER_BOX = 8


@wp.kernel
def _fill(quanta: wp.array4d(dtype=wp.int32), phase: wp.array4d(dtype=wp.float32), seed: int, largest: int):
    universe, ix, iy, iz = wp.tid()
    side = quanta.shape[1]
    state = wp.rand_init(seed, ((universe * side + ix) * side + iy) * side + iz)
    quanta[universe, ix, iy, iz] = wp.randi(state, -largest, largest + 1)
    phase[universe, ix, iy, iz] = (wp.randf(state) * 2.0 - 1.0) * wp.pi


def new_quanta_batch(universe_count: int, side: int, seed: int = 1, largest: int = 8,
                     device: str = "cuda") -> QuantaBatch:
    """A seeded batch of universe_count universes of side**3 boxes of uniform signed quanta and phase."""
    if universe_count < 1 or side < 2 or largest < 1:
        raise ValueError(f"need universe_count >= 1, side >= 2, largest >= 1, got {universe_count}, {side}, {largest}")
    boxes = universe_count * side ** 3
    if boxes >= 2 ** 31:
        raise ValueError(f"{boxes} boxes is past the int32 box index (2**31)")
    target = wp.get_device(device)
    if target.is_cuda and boxes * BYTES_PER_BOX > target.free_memory:
        raise MemoryError(f"{boxes} boxes need {boxes * BYTES_PER_BOX / 1e9:.2f} GB, "
                          f"{target.free_memory / 1e9:.2f} GB free on {target}")
    shape = (universe_count, side, side, side)
    quanta = wp.empty(shape, dtype=wp.int32, device=target)
    phase = wp.empty(shape, dtype=wp.float32, device=target)
    wp.launch(_fill, dim=shape, inputs=[quanta, phase, int(seed), int(largest)], device=target)
    return QuantaBatch(quanta, phase)
