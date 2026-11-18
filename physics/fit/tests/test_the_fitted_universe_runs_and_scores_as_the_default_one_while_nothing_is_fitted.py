"""While nothing is fitted, a universe run from the fitted settings is the default universe, number for number, and the same run twice gives the same record."""
import dataclasses

import physics.fit
from physics.engine import Settings
from physics.fit import load_fitted, run_universe
from physics.fit.tests.needs_gpu import needs_gpu

_HELD = {"push_strength": 40.0, "tear_length": 200.0}


def test_every_public_name_is_there():
    assert all(callable(getattr(physics.fit, name)) or name == "SEARCHED" for name in physics.fit.__all__)
    assert len(physics.fit.SEARCHED) == 12


@needs_gpu
def test_the_fitted_settings_change_nothing_while_nothing_is_fitted():
    fitted = load_fitted()["settings"]
    assert fitted == Settings()
    changed = {name: value for name, value in dataclasses.asdict(fitted).items() if value != getattr(Settings(), name)}
    default_run, fitted_run = run_universe({}, 1, 8, 24, 4), run_universe(changed, 1, 8, 24, 4)
    assert default_run["series"] == fitted_run["series"] and default_run["score"] == fitted_run["score"] == 0.0
    assert default_run["ended"] == "fewer than 5% of the start's volumes alive", "the default 8-a-side universe falls into one volume"


@needs_gpu
def test_the_same_run_twice_gives_the_same_series_and_score():
    first, second = run_universe(_HELD, 1, 8, 24, 4), run_universe(_HELD, 1, 8, 24, 4)
    assert first["series"] == second["series"] and first["score"] == second["score"]
    assert 0.0 < first["score"] < 1.0 and first["hold"] == 1.0 and first["series"]["step"] == [0, 4, 8, 12, 16, 20, 24]
