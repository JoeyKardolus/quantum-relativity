"""Clocks in the fullest energy volumes run slower than the emptiest volumes' clocks and than the step counter, and keep rising."""
import numpy as np

from physics.scenes.tests._gpu import HELD, needs_gpu
from physics.scenes.timescape.logic.timescape import timescape


@needs_gpu
def test_dense_clocks_run_behind_empty_clocks_which_run_behind_the_counter():
    result = timescape(8, steps=8, every=2, observers=2, settings=HELD)
    times, dense, empty = (np.array(result[name]) for name in ("time", "observer_time", "void_time"))
    print(f"after 8 steps (2.0 of counter time): fullest volumes' clocks {dense[-1]:.3f}, emptiest {empty[-1]:.3f}")
    assert result["observers"] == 2 and dense[0] == 0.0 and empty[0] == 0.0
    assert (np.diff(dense) > 0.0).all() and (np.diff(empty) > 0.0).all()
    assert dense[-1] < empty[-1] < times[-1] and result["clock_ratio"] < 1.0
    assert times.tolist() == [0.0, 0.5, 1.0, 1.5, 2.0]
