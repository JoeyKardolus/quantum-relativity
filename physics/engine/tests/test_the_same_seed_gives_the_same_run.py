"""The same seed gives the same run, box for box; another seed gives another."""
import numpy as np

from physics.engine import one_cell, step, three_cells


def _after(seed: int, steps: int, lay=one_cell) -> tuple[np.ndarray, np.ndarray]:
    store = lay(seed=seed)
    for _ in range(steps):
        step(store)
    return store.box_quanta.numpy(), store.box_phase.numpy()


def test_the_same_seed_gives_the_same_run():
    first, again, other = _after(1, 300), _after(1, 300), _after(2, 300)
    assert np.array_equal(first[0], again[0]) and np.array_equal(first[1], again[1])
    assert not np.array_equal(first[0], other[0])


def test_the_same_seed_gives_the_same_crossing():
    first, again, other = (_after(seed, 300, three_cells) for seed in (1, 1, 2))
    assert np.array_equal(first[0], again[0]) and np.array_equal(first[1], again[1])
    assert not np.array_equal(first[0], other[0])
