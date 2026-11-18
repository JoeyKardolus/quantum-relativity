"""Fitted settings load from a file into Settings with the other fields at their defaults, the defaults stay what they were, and a wrong name is refused."""
import dataclasses
import json

import pytest

from physics.engine import Settings
from physics.fit.fitted.logic.load_fitted import load_fitted


def test_a_fitted_file_gives_settings_without_touching_the_defaults(tmp_path):
    before = dataclasses.asdict(Settings())
    stored = tmp_path / "settings.json"
    stored.write_text(json.dumps({"settings": {"pull_strength": 3e-7, "tear_length": 14.0, "unevenness": 0.5},
                                  "fitted_on": {"seeds": [1, 2]}}))
    fitted = load_fitted(stored)
    assert fitted["settings"].pull_strength == 3e-7 and fitted["settings"].tear_length == 14.0
    assert fitted["settings"].unevenness == 0.5 and fitted["settings"].push_strength == before["push_strength"]
    assert fitted["fitted_on"] == {"seeds": [1, 2]}
    assert dataclasses.asdict(Settings()) == before


def test_a_name_that_is_no_setting_a_refused_value_and_a_stray_part_are_refused(tmp_path):
    stored = tmp_path / "settings.json"
    stored.write_text(json.dumps({"settings": {"dark_energy": 1.0}}))
    with pytest.raises(TypeError, match="dark_energy"):
        load_fitted(stored)
    stored.write_text(json.dumps({"settings": {"collapse_thin": 1.0}}))
    with pytest.raises(ValueError, match="collapse_thin"):
        load_fitted(stored)
    stored.write_text(json.dumps({"settings": {}, "start": {"unevenness": 0.5}}))
    with pytest.raises(ValueError, match="start"):
        load_fitted(stored)
    stored.write_text(json.dumps({"fitted_on": {}}))
    with pytest.raises(KeyError):
        load_fitted(stored)
    with pytest.raises(FileNotFoundError):
        load_fitted(tmp_path / "missing.json")


def test_a_file_with_no_values_is_the_default_universe(tmp_path):
    stored = tmp_path / "settings.json"
    stored.write_text(json.dumps({"settings": {}}))
    assert load_fitted(stored) == {"settings": Settings(), "fitted_on": {}}
