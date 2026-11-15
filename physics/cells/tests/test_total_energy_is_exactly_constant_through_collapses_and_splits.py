"""Total energy never changes: the leak moves it, a collapse hands it on, a split shares it out."""
import numpy as np

from physics.cells import EngineSettings, new_engine, step


def test_total_energy_stays_constant_over_100_steps_with_collapses_and_splits():
    """Pull at three times its default for collapses; from step 95 a split size just below the biggest volumes,
    so a few split each step."""
    engine = new_engine(200, 3, 0.5)
    start = engine.grid.total_energy()
    for index in range(100):
        biggest = float(np.sort(engine.grid.volume)[-3:].mean()) / engine.start_mean_volume
        engine = step(engine, EngineSettings.all_on(split_size=biggest if index >= 95 else np.inf,
                                                    attraction=0.015))
        assert np.isclose(engine.grid.total_energy(), start, rtol=1e-12, atol=0)
        assert (engine.grid.energy > 0).all()
    assert engine.collapses > 0 and engine.splits > 0
