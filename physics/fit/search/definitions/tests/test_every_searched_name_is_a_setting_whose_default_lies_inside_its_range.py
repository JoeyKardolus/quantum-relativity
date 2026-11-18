"""The twelve searched names are engine settings, each default inside its range, and no setting the engine lacks is searched."""
import dataclasses

from physics.engine import Settings
from physics.fit.search.definitions.ranges import LINEAR_RANGES, LOG_RANGES, SEARCHED

_NOT_SEARCHED = {"step_time", "size_constant", "slope", "inner_unevenness", "slowest_speed", "fastest_speed"}


def test_the_searched_names_and_the_ones_left_alone_are_all_of_the_engines_settings():
    fields = {field.name for field in dataclasses.fields(Settings)}
    assert len(SEARCHED) == len(set(SEARCHED)) == 12 and SEARCHED == list(LOG_RANGES) + list(LINEAR_RANGES)
    assert set(SEARCHED) | _NOT_SEARCHED == fields and not set(SEARCHED) & _NOT_SEARCHED


def test_every_default_lies_inside_its_range_and_every_range_runs_upward():
    defaults = Settings()
    for name, (low, high) in (LOG_RANGES | LINEAR_RANGES).items():
        assert low < high and low <= getattr(defaults, name) <= high, name
    assert all(low > 0.0 for low, _ in LOG_RANGES.values()), "a logarithm needs a range above 0"


def test_both_ends_of_every_range_are_settings_the_engine_accepts():
    for end in (0, 1):
        Settings(**{name: limits[end] for name, limits in (LOG_RANGES | LINEAR_RANGES).items()})
