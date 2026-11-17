"""The fitted settings load from their own file into Settings and start arguments, and the defaults stay what they were."""
import dataclasses
import json

import pytest

from physics.engine import Settings
from physics.fit import load_fitted
from physics.fit.fit_search import LINEAR_RANGES, LOG_RANGES, START_RANGES


def test_a_fitted_file_gives_settings_and_start_without_touching_the_defaults(tmp_path):
    before = dataclasses.asdict(Settings())
    stored = tmp_path / "settings.json"
    stored.write_text(json.dumps({"settings": {"pull_strength": 3e-7, "split_across": 14.0},
                                  "start": {"unevenness": 0.5}, "fitted_on": {"seeds": [1, 2]}}))
    fitted = load_fitted(stored)
    assert fitted["settings"].pull_strength == 3e-7 and fitted["settings"].split_across == 14.0
    assert fitted["settings"].push == before["push"]
    assert fitted["start"] == {"unevenness": 0.5} and fitted["fitted_on"] == {"seeds": [1, 2]}
    assert dataclasses.asdict(Settings()) == before


def test_a_name_that_is_no_setting_is_refused(tmp_path):
    stored = tmp_path / "settings.json"
    stored.write_text(json.dumps({"settings": {"dark_energy": 1.0}, "start": {}}))
    with pytest.raises(TypeError):
        load_fitted(stored)
    stored.write_text(json.dumps({"settings": {}, "start": {"seed": 3}}))
    with pytest.raises(ValueError):
        load_fitted(stored)


def test_the_shipped_fitted_file_lies_inside_the_searched_ranges():
    fitted = load_fitted()
    ranges = LOG_RANGES | LINEAR_RANGES
    for name, (low, high) in ranges.items():
        assert low <= getattr(fitted["settings"], name) <= high
    for name, value in fitted["start"].items():
        assert START_RANGES[name][0] <= value <= START_RANGES[name][1]
    assert fitted["fitted_on"]["seeds"]
