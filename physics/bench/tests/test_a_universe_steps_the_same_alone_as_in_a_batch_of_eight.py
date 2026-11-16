"""Universes side by side never touch: one universe stepped alone matches its twin in a batch of 8.

The batch dimension keeps every neighbour lookup inside its own universe, so universe 5
of a batch of 8 (16**3 boxes, seed 3) and a batch of one started from its copy end
equal after 30 diffusion steps: the quanta to the quantum, the phase to float32.
"""
import numpy as np
import warp as wp

from physics.bench import QuantaBatch, diffusion_step, new_quanta_batch
from physics.bench.tests._gpu import needs_gpu


@needs_gpu
def test_universe_5_alone_and_in_the_batch_end_equal():
    batch = new_quanta_batch(8, 16, seed=3)
    alone = QuantaBatch(wp.array(batch.quanta.numpy()[5:6], dtype=wp.int32, device="cuda"),
                        wp.array(batch.phase.numpy()[5:6], dtype=wp.float32, device="cuda"))
    for _ in range(30):
        diffusion_step(batch)
        diffusion_step(alone)
    assert np.array_equal(batch.quanta.numpy()[5], alone.quanta.numpy()[0])
    assert np.allclose(batch.phase.numpy()[5], alone.phase.numpy()[0], atol=1e-6)
    assert not np.array_equal(batch.quanta.numpy()[5], batch.quanta.numpy()[4])
