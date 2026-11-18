"""The streaks are the tails oldest mark first, each mark as solid as its tracer went fast to it."""
import numpy as np
import warp as wp

from physics.view.logic.streaks import streaks
from physics.view.logic.tracers import Tracers


def test_the_marks_follow_each_other_from_the_oldest_to_now():
    way = np.array([[0.0, 0, 0], [0.3, 0, 0], [0.3, 0.5, 0], [0.3, 0.5, 12]], dtype=np.float32)
    for head in range(4):  # the same way, kept from any row of the ring on
        ring = np.roll(way, head + 1 - 4, axis=0)[:, None, :].repeat(2, axis=1)
        drawn = streaks(Tracers(wp.array(ring, dtype=wp.vec3, device="cpu"), head)).numpy()
        assert drawn.shape == (4, 2, 4) and np.array_equal(drawn[:, 1, :3], way)
        assert np.allclose(drawn[:, 0, 3], [0, 0.25, 0.25 / 0.36, 1])  # the way squared over FULL's


def test_a_tracer_at_rest_is_not_drawn():
    still = np.full((5, 3, 3), 7.5, dtype=np.float32)
    assert streaks(Tracers(wp.array(still, dtype=wp.vec3, device="cpu"), 2)).numpy()[..., 3].max() == 0
