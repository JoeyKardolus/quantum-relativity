"""Clocks in the most packed energy volumes run slower than the step counter, and keep rising (time is local)."""
import numpy as np

from physics.engine import Settings, new_universe
from physics.scenes import clocked_run, observer_cells
from physics.scenes.tests._gpu import needs_gpu


@needs_gpu
def test_observers_in_the_densest_lumps_keep_slower_local_time_than_the_counter():
    settings = Settings()
    universe = new_universe(6, 4, seed=1)
    samples, local = clocked_run(universe, settings, steps=6, every=2, measure=lambda sampled: None)
    picked = observer_cells(universe, 2, local)
    times = np.array([when for when, _ in samples])
    assert picked.size == 2
    assert local.shape[0] == times.size and local.shape[1] >= picked.size
    observer_time = local[:, picked]
    assert (observer_time[0] == 0.0).all()
    assert (np.diff(observer_time, axis=0) > 0.0).all()
    assert (observer_time[-1] < times[-1]).all()
