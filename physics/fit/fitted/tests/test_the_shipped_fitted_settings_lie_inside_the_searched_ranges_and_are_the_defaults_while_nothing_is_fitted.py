"""What the package ships as fitted names only searched settings inside their ranges, and is the default universe while nothing is fitted."""
from physics.engine import Settings
from physics.fit.fitted.definitions.fitted_values import FITTED
from physics.fit.fitted.logic.load_fitted import load_fitted
from physics.fit.search.definitions.ranges import LINEAR_RANGES, LOG_RANGES


def test_every_shipped_value_was_searched_and_lies_inside_its_range():
    fitted = load_fitted()
    ranges = LOG_RANGES | LINEAR_RANGES
    for name in FITTED["settings"]:
        assert ranges[name][0] <= getattr(fitted["settings"], name) <= ranges[name][1]
    assert fitted["fitted_on"] == FITTED["fitted_on"] and fitted["fitted_on"] is not FITTED["fitted_on"]


def test_with_nothing_fitted_the_fitted_universe_is_the_default_one():
    assert FITTED["settings"] == {}, "no search has run on the rebuilt engine"
    assert load_fitted()["settings"] == Settings()
