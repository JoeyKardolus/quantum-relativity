"""A start that doubles the volumes' energy gives a universe whose volumes hold twice as much, and its run keeps exact books."""
import numpy as np

from physics.engine import ENERGY_VOLUME, book_law, new_universe, read_books
from physics.scenes.run import even_volume_quanta, sampled_run, start_settings
from physics.scenes.tests._gpu import HELD, needs_gpu


@needs_gpu
def test_doubled_volume_quanta_in_a_start_doubles_what_the_volumes_hold():
    settings = start_settings(HELD, {"volume_quanta": 2.0 * HELD.volume_quanta})
    universe = new_universe(settings, 4, seed=1)
    held = universe.host("cell_quanta")[universe.host("kind") == ENERGY_VOLUME]
    assert abs(held.mean() / even_volume_quanta(HELD) - 2.0) < 0.04
    assert abs(held.mean() / even_volume_quanta(settings) - 1.0) < 0.02
    samples = sampled_run(universe, settings, 2, 1, lambda sampled: book_law(sampled.start_books, read_books(sampled)))
    assert all(set(gap.values()) == {0} for _, gap in samples), "every change of the total is booked"
