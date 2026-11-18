"""Every definition of the largest universe on numbers worked by hand."""
import pytest

from physics.bench.largest_universe.definitions.limits import MEMORY, SIZE_LIST, SPEED, limit_law
from physics.bench.largest_universe.definitions.memory_law import BYTES_PER_CELL_ID, bytes_needed, fits
from physics.bench.largest_universe.definitions.pass_mark import MIN_RATE, check_rate, meets_rate
from physics.bench.largest_universe.definitions.search import Probe
from physics.bench.largest_universe.definitions.size_ladder import SIZES, check_sizes


def test_the_mark_is_one_step_a_second_and_at_the_mark_passes():
    assert MIN_RATE == 1.0
    assert meets_rate(1.0) and meets_rate(1.01) and not meets_rate(0.99)
    assert meets_rate(2.9, min_rate=3.0) is False
    for rate in (0.0, -1.0):
        with pytest.raises(ValueError):
            check_rate(rate)


def test_ten_a_side_with_a_quarter_spare_is_1250_ids():
    assert bytes_needed(10, spare=0.25) == 1250 * BYTES_PER_CELL_ID
    assert bytes_needed(10, spare=0.0) == 1000 * BYTES_PER_CELL_ID
    assert fits(10, 1250 * BYTES_PER_CELL_ID, spare=0.25)
    assert not fits(10, 1250 * BYTES_PER_CELL_ID - 1, spare=0.25)


def test_the_ladder_is_even_rising_and_starts_at_the_smallest_lattice():
    assert SIZES[0] == 4 and SIZES[-1] == 64 and len(SIZES) == 31
    assert all(size % 2 == 0 for size in SIZES)
    check_sizes(SIZES)
    for bad in ([], [8, 4], [4, 4]):
        with pytest.raises(ValueError):
            check_sizes(bad)


def test_speed_is_named_before_memory_and_the_list_only_when_nothing_failed():
    fast, slow, big = Probe(4, True, 64, 5.0), Probe(8, True, 512, 0.5), Probe(16, False)
    assert limit_law([fast, slow, big], 1.0) == SPEED
    assert limit_law([fast, big], 1.0) == MEMORY
    assert limit_law([fast], 1.0) == SIZE_LIST
    assert limit_law([fast], 6.0) == SPEED
