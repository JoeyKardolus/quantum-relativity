"""The live Simulation steps a cell grid, a reset is reproducible by seed, and positions never reach the Grid."""
from dataclasses import fields

import numpy as np
import pytest

pytest.importorskip("physics.cells", reason="physics.cells (the cell grid) is not built yet")

from physics.app import Simulation  # noqa: E402
from physics.app.cell_layout import cell_layout  # noqa: E402
from physics.cells import Grid  # noqa: E402

CELLS = 300


def test_the_simulation_steps_and_a_reset_is_reproducible_by_seed():
    first, second = Simulation(CELLS, seed=4), Simulation(CELLS, seed=4)
    start_volume = float(first.grid.volume.sum())
    for _ in range(3):
        first.advance()
        second.advance()
    assert first.steps == 3
    assert float(first.grid.volume.sum()) != start_volume      # pull and the leak are always on
    assert np.array_equal(first.grid.energy, second.grid.energy)
    assert np.array_equal(first.grid.volume, second.grid.volume)
    assert np.allclose(first.layout(), second.layout())
    first.reset(5, CELLS)
    assert first.steps == 0 and first.start_cells == first.grid.cell_count


def test_display_positions_follow_every_step_and_never_end_up_on_the_grid():
    simulation = Simulation(CELLS, seed=3)
    simulation.apply({"push_limit": 0.99})     # every squeezed joint gives way: collapses happen
    for _ in range(5):
        simulation.advance()
        positions = simulation.layout()
        assert positions.shape == (simulation.grid.cell_count, 3)
        assert np.all(np.isfinite(positions))
        assert simulation.radius().shape == (simulation.grid.cell_count,)
    for field in fields(Grid):
        value = getattr(simulation.grid, field.name)
        assert not (isinstance(value, np.ndarray) and value.ndim == 2 and value.shape[1] == 3)


def test_the_layout_puts_neighbours_at_their_radius_sum_and_skips_wrap_around_joints():
    positions = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [40.0, 0.0, 0.0]])
    joints = np.array([[0, 1], [1, 2]])
    radius = np.array([1.0, 2.0, 1.0])
    relaxed = cell_layout(positions, joints, radius, 60)
    assert np.linalg.norm(relaxed[0] - relaxed[1]) == pytest.approx(3.0, rel=1e-3)
    assert np.array_equal(relaxed[2], positions[2])      # its one joint wraps around: left alone


def test_settings_are_clamped_to_their_range_and_unknown_names_refused():
    simulation = Simulation(CELLS, seed=1)
    simulation.apply({"leak": 7.0, "speed": 0.0})
    assert simulation.settings.leak == simulation.ranges["leak"][1] <= 1.0
    assert simulation.speed == simulation.ranges["speed"][0]
    with pytest.raises(ValueError):
        simulation.apply({"gravity": 1.0})
