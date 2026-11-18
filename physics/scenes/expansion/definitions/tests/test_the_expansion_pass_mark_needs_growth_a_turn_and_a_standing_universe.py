"""The mark passes only a universe that grew 5%, slowed then sped up, and stands at both lengths."""
from physics.scenes.expansion.definitions.pass_mark import passes

GOOD = {"size_growth": [1.0, 1.03, 1.06], "size_rate_trend_early": -0.01, "size_rate_trend_late": 0.02}


def test_the_hoped_for_shape_passes_when_the_universe_stands():
    assert passes(GOOD, True, True)


def test_each_missing_part_is_a_miss():
    assert not passes(GOOD, True, False) and not passes(GOOD, False, True)
    assert not passes({**GOOD, "size_growth": [1.0, 1.04]}, True, True), "grew under 5%"
    assert not passes({**GOOD, "size_rate_trend_early": 0.01}, True, True), "never slowed"
    assert not passes({**GOOD, "size_rate_trend_late": -0.02}, True, True), "never sped up"
    assert not passes({**GOOD, "size_rate_trend_late": float("nan")}, True, True)
