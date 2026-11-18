"""A universe run from the fitted settings stands like the default one, and the same run twice gives the same record."""
import dataclasses

import physics.fit
from physics.engine import Settings
from physics.fit import load_fitted, run_universe
from physics.fit.tests.needs_gpu import needs_gpu

_HELD = {"push_strength": 40.0, "tear_length": 200.0}


def test_every_public_name_is_there():
    assert all(callable(getattr(physics.fit, name)) or name == "SEARCHED" for name in physics.fit.__all__)
    assert len(physics.fit.SEARCHED) == 10


@needs_gpu
def test_the_fitted_settings_are_the_defaults_but_for_what_phase_1_names_and_their_universe_runs():
    fitted = load_fitted()["settings"]
    changed = {name: value for name, value in dataclasses.asdict(fitted).items() if value != getattr(Settings(), name)}
    assert set(changed) <= set(load_fitted()["fitted_on"]["names"])
    default_run, fitted_run = run_universe({}, 1, 8, 24, 4), run_universe(changed, 1, 8, 24, 4)
    assert default_run["ended"] == "", "the default 8-a-side universe stands"
    assert fitted_run["steps_done"] == 24 and fitted_run["ended"] == "", "and so does the fitted one"


@needs_gpu
def test_the_same_run_twice_gives_the_same_series_and_score():
    first, second = run_universe(_HELD, 1, 8, 24, 4), run_universe(_HELD, 1, 8, 24, 4)
    assert first["series"] == second["series"] and first["score"] == second["score"]
    assert 0.0 <= first["score"] < 1.0 and first["hold"] == 1.0 and first["series"]["step"] == [0, 4, 8, 12, 16, 20, 24]
