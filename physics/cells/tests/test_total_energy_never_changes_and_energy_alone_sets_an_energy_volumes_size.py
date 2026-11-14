"""Total energy never changes, and an energy volume's size is the one its energy gives.

Bible rule 8: a push moves no energy, so energy moves only by the leak and total
energy never changes; rule 4: an energy volume's size follows only its energy.
For 200 full engine steps with every rule on at the defaults (300 volumes, seed 4,
unevenness 0.1), after every step: total energy equals the start's to rounding
(rtol 1e-12; the leak and collapse only move and add energy), and every energy
volume's size is exactly rest_volume of its energy.
"""
from __future__ import annotations

import numpy as np

from physics.cells import EngineSettings, new_engine, rest_volume, step

STEPS = 200


def test_total_energy_constant_and_size_from_energy_every_step():
    engine = new_engine(300, seed=4, unevenness=0.1)
    settings = EngineSettings.all_on()
    start = engine.grid.total_energy()
    for _ in range(STEPS):
        engine = step(engine, settings)
        assert np.isclose(engine.grid.total_energy(), start, rtol=1e-12, atol=0)
        assert np.array_equal(engine.grid.volume, rest_volume(engine.grid.energy))
