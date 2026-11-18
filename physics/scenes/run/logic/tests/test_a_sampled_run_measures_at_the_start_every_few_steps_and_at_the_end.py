"""Five steps sampled every two are measured at steps 0, 2, 4 and 5, at the counter's time."""
import pytest

from physics.engine import new_universe
from physics.scenes.run.logic.sampled_run import sampled_run
from physics.scenes.tests._gpu import HELD, needs_gpu


@needs_gpu
def test_samples_fall_at_the_start_every_second_step_and_the_last():
    universe = new_universe(HELD, 4, seed=1)
    samples = sampled_run(universe, HELD, 5, 2, lambda sampled: sampled.step)
    assert [taken for _, taken in samples] == [0, 2, 4, 5]
    assert [time for time, _ in samples] == [0.0, 0.5, 1.0, 1.25]
    assert universe.step == 5, "the universe itself was stepped"


@needs_gpu
def test_a_run_of_no_steps_measures_the_start_once_and_bad_counts_raise():
    universe = new_universe(HELD, 4, seed=1)
    assert sampled_run(universe, HELD, 0, 1, lambda sampled: sampled.step) == [(0.0, 0)]
    with pytest.raises(ValueError):
        sampled_run(universe, HELD, 3, 0, lambda sampled: 0)
    with pytest.raises(ValueError):
        sampled_run(universe, HELD, -1, 1, lambda sampled: 0)
