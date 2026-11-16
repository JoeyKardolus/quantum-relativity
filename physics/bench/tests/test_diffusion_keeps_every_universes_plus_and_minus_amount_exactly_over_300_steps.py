"""The trivial diffusion step keeps each universe's plus and minus amount exactly while it smooths.

Bible rule 4: plus and minus never cancel and the amount is whole, so both totals hold
to the quantum, per universe, in a batch of 8 universes of 16**3 boxes, seed 2.
The step must also do something: same-sign pairs share, so the spread of the plus
boxes' amounts falls.
"""
import numpy as np

from physics.bench import amounts, diffusion_step, new_quanta_batch
from physics.bench.tests._gpu import needs_gpu


@needs_gpu
def test_plus_and_minus_amounts_hold_exactly_and_the_quanta_even_out():
    batch = new_quanta_batch(8, 16, seed=2)
    plus_before, minus_before = amounts(batch)
    spread_before = np.std(batch.quanta.numpy()[batch.quanta.numpy() > 0])
    for _ in range(300):
        diffusion_step(batch)
    plus_after, minus_after = amounts(batch)
    quanta = batch.quanta.numpy()
    spread_after = np.std(quanta[quanta > 0])
    print(f"plus {plus_before.tolist()} -> {plus_after.tolist()}; minus {minus_before.tolist()} -> "
          f"{minus_after.tolist()}; spread of plus boxes {spread_before:.3f} -> {spread_after:.3f}")
    assert batch.steps == 300
    assert np.array_equal(plus_before, plus_after)
    assert np.array_equal(minus_before, minus_after)
    assert spread_after < spread_before
