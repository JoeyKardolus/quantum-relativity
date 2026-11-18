"""The source interface and the stat names."""
from physics.view.source.definitions.source_interface import Source
from physics.view.source.definitions.stat_names import STAT_NAMES
from physics.view.tests.grid_cases import FakeSource


def test_a_plain_array_double_fits_and_one_that_cannot_be_drawn_does_not():
    assert isinstance(FakeSource(), Source)

    class NoView:
        steps = 0

        def reset(self): ...
        def step(self): ...
        def stats(self): ...

    assert not isinstance(NoView(), Source)


def test_the_title_reads_nine_counts_each_once():
    assert len(STAT_NAMES) == len(set(STAT_NAMES)) == 9
    assert {"volumes", "joints", "collapses", "tears", "endless", "size", "size_growth", "amount", "warp"} == set(STAT_NAMES)
