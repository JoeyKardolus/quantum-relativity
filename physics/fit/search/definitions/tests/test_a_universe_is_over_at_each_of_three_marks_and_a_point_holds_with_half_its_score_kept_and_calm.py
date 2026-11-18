"""A universe is over past two tears a start volume, under 5% of its volumes alive or over 95% spent; a point holds with half its short score kept and a hold above 0.9."""
from physics.fit.search.definitions.point_holds import CALM, HALF_KEPT, point_holds
from physics.fit.search.definitions.torn_out import torn_out
from physics.fit.search.definitions.why_ended import NOTHING_LEFT, why_ended


def test_a_standing_universe_has_not_ended_and_each_mark_is_passed_only_beyond_it():
    assert why_ended(128, 64, 0.05, 0.95) == "", "exactly at all three marks it still stands"
    assert why_ended(129, 64, 1.0, 0.0) == "more than two tears per start volume"
    assert why_ended(0, 64, 0.049, 0.0) == "fewer than 5% of the start's volumes alive"
    assert why_ended(0, 64, 1.0, 0.951) == "more than 95% of the size spent"
    assert why_ended(129, 64, 0.0, 1.0) == "more than two tears per start volume", "the tear mark is named first"
    assert not torn_out(128, 64) and torn_out(129, 64)
    assert NOTHING_LEFT == "no energy volume holds any energy"


def test_a_point_holds_only_with_half_its_score_kept_and_a_calm_long_run():
    assert (HALF_KEPT, CALM) == (0.5, 0.9)
    assert point_holds(0.2, 0.1, 0.95) and point_holds(0.2, 0.3, 1.0)
    assert not point_holds(0.2, 0.099, 0.95), "less than half kept"
    assert not point_holds(0.2, 0.2, 0.9), "not calm: the hold factor must be above 0.9"
    assert not point_holds(0.0, 0.0, 1.0), "a point that scored nothing never holds"
