"""What the package ships as fitted names only searched settings inside their ranges, and is the default universe while nothing is fitted."""
from physics.engine import Settings
from physics.fit.fitted.definitions.fitted_values import FITTED
from physics.fit.fitted.logic.load_fitted import load_fitted
from physics.fit.search.definitions.gravity_ranges import GRAVITY_RANGES


def test_every_shipped_value_was_searched_and_lies_inside_its_range():
    fitted = load_fitted()
    ranges = GRAVITY_RANGES     # phase 1 of the fit searches only gravity's settings
    for name in FITTED["settings"]:
        assert ranges[name][0] <= getattr(fitted["settings"], name) <= ranges[name][1]
    assert fitted["fitted_on"] == FITTED["fitted_on"] and fitted["fitted_on"] is not FITTED["fitted_on"]


def test_what_is_fitted_is_marked_with_its_phase_and_differs_from_the_defaults_only_in_what_it_names():
    assert FITTED["fitted_on"]["phase"] == "phase 1: gravity"
    fitted, default = load_fitted()["settings"], Settings()
    for field in fitted.__dataclass_fields__:
        if field not in FITTED["settings"]:
            assert getattr(fitted, field) == getattr(default, field), field
