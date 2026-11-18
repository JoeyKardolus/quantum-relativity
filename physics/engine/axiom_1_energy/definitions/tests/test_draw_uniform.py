"""The draw repeats for the same key, changes with any part of it, and lies in [0, 1)."""
import numpy as np
import warp as wp

from physics.engine.axiom_1_energy.definitions.draw_uniform import draw_uniform


@wp.kernel
def _draws(seed: wp.uint32, step: wp.uint32, pass_id: wp.uint32, out: wp.array(dtype=wp.float64)):
    out[wp.tid()] = draw_uniform(seed, step, pass_id, wp.uint64(wp.tid()))


def _row(seed: int, step: int, pass_id: int) -> np.ndarray:
    out = wp.zeros(4096, dtype=wp.float64, device="cpu")
    wp.launch(_draws, dim=4096, inputs=[seed, step, pass_id, out], device="cpu")
    return out.numpy()


def test_the_draw_repeats_and_lies_between_0_and_1():
    first = _row(1, 5, 2)
    assert np.array_equal(first, _row(1, 5, 2))
    assert first.min() >= 0.0 and first.max() < 1.0 and abs(first.mean() - 0.5) < 0.02
    assert len(np.unique(first)) == 4096
    for other in (_row(2, 5, 2), _row(1, 6, 2), _row(1, 5, 3)):
        assert not np.array_equal(first, other)
