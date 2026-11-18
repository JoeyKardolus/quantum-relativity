"""The orbit start differs from the same start without a ramp only in the small volume's phases, by exactly the ramp."""
import numpy as np
import pytest

from physics.scenes.orbit.definitions.sideways_ramp import SIDEWAYS
from physics.scenes.orbit.logic.orbit_start import orbit_start
from physics.scenes.tests._gpu import HELD, needs_gpu


@needs_gpu
def test_the_small_volume_gets_the_ramp_and_nothing_else_changes():
    moving, twin, small, middle = orbit_start(16, 2, 8.0, 2.0, SIDEWAYS, settings=HELD)
    resting, _, _, _ = orbit_start(16, 2, 8.0, 2.0, 0.0, settings=HELD)
    assert small == (6, 4, 4) and middle == 4
    small_cell, big_cell = (12 * 16 + 8) * 16 + 8, (8 * 16 + 8) * 16 + 8
    turn = moving.block("phase", small_cell) - resting.block("phase", small_cell)
    ramp = SIDEWAYS * np.arange(6).reshape(1, 6, 1)
    assert np.exp(1j * turn) == pytest.approx(np.exp(1j * np.broadcast_to(ramp, (6, 6, 6))))
    assert np.array_equal(moving.block("phase", big_cell), resting.block("phase", big_cell))
    assert np.array_equal(moving.host("cell_quanta"), resting.host("cell_quanta"))
    assert twin.host("cell_quanta")[big_cell] == 8 * 648000 and moving.host("cell_quanta")[small_cell] == 2 * 648000
    differing = np.flatnonzero(twin.host("cell_quanta") != moving.host("cell_quanta"))
    assert differing.tolist() == [small_cell], "the twin is the same start but for the small volume"
    with pytest.raises(ValueError):
        orbit_start(16, 4, 8.0, 2.0, SIDEWAYS, settings=HELD)
