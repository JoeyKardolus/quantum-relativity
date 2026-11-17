"""Clocks in the most packed energy volumes run slower than the step counter and than the emptiest volumes' clocks, and keep rising (time is local)."""
import numpy as np

from physics.engine import Settings, new_universe
from physics.scenes import observer_cells, timescape
from physics.scenes.tests._gpu import needs_gpu


@needs_gpu
def test_observers_in_the_densest_lumps_keep_slower_local_time_than_the_counter():
    result = timescape(8, steps=8, every=2, observers=2)
    times, dense, empty = (np.array(result[name]) for name in ("time", "observer_time", "void_time"))
    assert result["observers"] == 2 and dense[0] == 0.0 and empty[0] == 0.0
    assert (np.diff(dense) > 0.0).all() and (np.diff(empty) > 0.0).all()
    assert dense[-1] < empty[-1] < times[-1] and result["clock_ratio"] < 1.0


@needs_gpu
def test_the_observers_are_the_volumes_with_the_most_quanta_most_packed_first():
    universe = new_universe(8, seed=1)
    picked = observer_cells(universe, 3)
    held = universe.host("cell_quanta")
    volumes = np.flatnonzero(universe.kind_h[:universe.count] == 1)
    assert picked.tolist() == volumes[np.argsort(-held[volumes], kind="stable")][:3].tolist()
    assert held[picked[0]] >= held[picked[1]] >= held[picked[2]] and Settings().clock_packing > 0
