"""The bisection on fakes with a known answer: a step that sleeps by size, and a build that stops fitting."""
import time

import pytest

from physics.bench.largest_universe.definitions.limits import MEMORY, SIZE_LIST, SPEED
from physics.bench.largest_universe.logic.largest_universe import largest_universe
from physics.bench.largest_universe.logic.probe_size import probe_size
from physics.bench.throughput.definitions.run import Run


def _sleeper(size):
    return Run(lambda: time.sleep(0.1 * size), size ** 3)


def _fits_to_five(size):
    if size > 5:
        raise MemoryError("too big")
    return Run(lambda: None, size, lambda: {"size_built": size})


def test_a_step_of_a_tenth_of_a_second_a_size_unit_passes_three_a_second_up_to_size_two():
    search = largest_universe(_sleeper, [1, 2, 4, 8, 16], min_rate=3.0, steps=1, warmup=0)
    assert (search.size, search.cells, search.limit) == (2, 8, SPEED)
    assert 3.0 <= search.steps_per_second <= 5.0
    assert [probe.size for probe in search.probes] == [4, 1, 2]


def test_a_build_that_stops_fitting_past_five_gives_five_and_names_memory():
    search = largest_universe(_fits_to_five, list(range(1, 11)), steps=1, warmup=0)
    assert (search.size, search.cells, search.limit) == (5, 5, MEMORY)
    assert [(probe.size, probe.fits) for probe in search.probes] == [(5, True), (8, False), (6, False)]


def test_when_every_size_passes_the_list_is_the_limit_and_when_none_does_there_is_no_size():
    every = largest_universe(_fits_to_five, [1, 2, 3], steps=1, warmup=0)
    assert (every.size, every.limit) == (3, SIZE_LIST)
    never = largest_universe(lambda size: Run(lambda: time.sleep(0.2), size), [1, 2], min_rate=100.0, steps=1,
                             warmup=0)
    assert (never.size, never.cells, never.steps_per_second, never.limit) == (None, None, None, SPEED)


def test_a_bad_list_or_rate_builds_nothing():
    built = []
    for sizes, rate in (([], 1.0), ([2, 1], 1.0), ([1, 2], 0.0)):
        with pytest.raises(ValueError):
            largest_universe(lambda size: built.append(size), sizes, min_rate=rate)
    assert built == []


def test_a_probe_reads_the_run_s_facts_and_only_a_memory_error_means_it_does_not_fit():
    probe = probe_size(_fits_to_five, 3, steps=2, warmup=0)
    assert (probe.size, probe.fits, probe.cells, probe.facts) == (3, True, 3, {"size_built": 3})
    refused = probe_size(_fits_to_five, 6, steps=2, warmup=0)
    assert (refused.fits, refused.cells, refused.steps_per_second) == (False, None, None)

    def broken(size):
        raise RuntimeError("the build broke")

    with pytest.raises(RuntimeError):
        probe_size(broken, 3, steps=1, warmup=0)
