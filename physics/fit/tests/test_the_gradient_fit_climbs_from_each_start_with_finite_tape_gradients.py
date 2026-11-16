"""The gradient fit takes finite tape gradients for every searched setting and moves each start inside the unit cube; a stand-in universe scores the same alone as side by side."""
import numpy as np
import warp as wp

from physics.engine import host_boxes, new_universe
from physics.fit import gradient_fit, stand_in_score
from physics.fit.fit_search import NAMES
from physics.scenes.tests._gpu import needs_gpu


@needs_gpu
def test_the_gradient_fit_climbs_from_each_start_with_finite_tape_gradients():
    start = host_boxes(new_universe(3, 4, seed=1, margin=0))["quanta"]
    starts = np.array([[0.5] * len(NAMES), [0.2, 0.8, 0.3, 0.6, 0.4, 0.7]])
    records = gradient_fit(start, starts, iterations=2, rate=0.05, steps=10, every=2, block=4)
    assert [record["iteration"] for record in records] == [0, 1, 2]
    for record in records[:-1]:
        gradient = np.asarray(record["gradient"])
        assert gradient.shape == starts.shape and np.isfinite(gradient).all()
        assert (np.abs(gradient) > 0).all(axis=1).any()
    last = np.asarray(records[-1]["units"])
    assert ((last >= 0) & (last <= 1)).all() and not np.allclose(last, starts)
    assert np.isfinite(records[-1]["score"]).all()
    alone, _ = stand_in_score(start, wp.array(starts[1:].astype(np.float32)), 10, 2, 4)
    together, _ = stand_in_score(start, wp.array(starts.astype(np.float32)), 10, 2, 4)
    assert abs(float(alone.numpy()[0]) - float(together.numpy()[1])) < 1e-3
