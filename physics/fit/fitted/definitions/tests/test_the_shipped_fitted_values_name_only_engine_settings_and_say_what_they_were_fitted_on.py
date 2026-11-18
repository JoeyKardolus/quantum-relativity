"""The shipped fitted values name only fields of the engine's Settings and always say what search they came from."""
import dataclasses

from physics.engine import Settings
from physics.fit.fitted.definitions.fitted_values import FITTED


def test_the_record_has_its_two_parts_and_only_real_setting_names():
    assert set(FITTED) == {"settings", "fitted_on"}
    assert set(FITTED["settings"]) <= {field.name for field in dataclasses.fields(Settings)}
    assert FITTED["fitted_on"]["status"] and FITTED["fitted_on"]["date"] and FITTED["fitted_on"]["search"]


def test_with_no_values_the_status_says_nothing_is_fitted():
    assert bool(FITTED["settings"]) != FITTED["fitted_on"]["status"].startswith("none")
