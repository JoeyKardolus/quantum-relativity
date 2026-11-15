"""Total energy never changes: the leak moves it, a collapse hands it on, a split shares it out."""
import numpy as np

from physics.cells import EngineSettings, new_engine, step


def test_total_energy_stays_constant_over_100_steps_with_collapses_and_splits():
    """Pull at three times its default for collapses; from step 95 a split threshold just above the thinnest
    volumes (a split keeps density, so split children split again each step: the split steps stay few)."""
    engine = new_engine(200, 3, 0.5)
    start = engine.grid.total_energy()
    for index in range(100):
        density = np.sort(engine.grid.energy / engine.grid.volume)
        threshold = float(density[:3].mean()) if index >= 95 else 0.0
        engine = step(engine, EngineSettings.all_on(split_threshold=threshold, attraction=0.015))
        assert np.isclose(engine.grid.total_energy(), start, rtol=1e-12, atol=0)
        assert (engine.grid.energy > 0).all()
    assert engine.collapses > 0 and engine.splits > 0
