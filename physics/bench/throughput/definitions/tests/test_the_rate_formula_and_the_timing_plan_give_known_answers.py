"""The rate formula on numbers worked by hand, and the plans a timing refuses."""
import dataclasses

import pytest

from physics.bench.throughput.definitions.rates import Rates, rates_law
from physics.bench.throughput.definitions.run import Run
from physics.bench.throughput.definitions.timing import check_timing


def test_ten_steps_of_a_thousand_cells_in_two_seconds():
    assert rates_law(1000, 10, 2.0) == Rates(cells=1000, steps=10, seconds=2.0, steps_per_second=5.0,
                                             cell_steps_per_second=5000.0)


def test_a_timing_that_took_no_time_has_no_rate():
    for seconds in (0.0, -1.0):
        with pytest.raises(ValueError):
            rates_law(1000, 10, seconds)


def test_a_plan_without_a_timed_step_a_negative_warmup_or_no_cells_is_refused():
    check_timing(cells=1, steps=1, warmup=0)
    for cells, steps, warmup in ((1, 0, 0), (1, 1, -1), (0, 1, 0)):
        with pytest.raises(ValueError):
            check_timing(cells, steps, warmup)


def test_a_run_has_no_facts_unless_given_and_cannot_be_changed():
    run = Run(advance=lambda: None, cells=8)
    assert run.facts() == {}
    with pytest.raises(dataclasses.FrozenInstanceError):
        run.cells = 9
