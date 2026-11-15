"""Total energy never changes: the leak moves it, a collapse adds it together, a split shares it out."""
import numpy as np

from physics.cells import EngineSettings, new_engine, step


def test_total_energy_stays_constant_over_200_steps_with_collapses_and_splits():
    engine = new_engine(200, 3, 0.5)
    settings = EngineSettings.all_on(split_threshold=0.3)
    start = engine.grid.total_energy()
    for _ in range(200):
        engine = step(engine, settings)
        assert np.isclose(engine.grid.total_energy(), start, rtol=1e-12, atol=0)
        assert (engine.grid.energy > 0).all()
    assert engine.collapses > 0 and engine.splits > 0
