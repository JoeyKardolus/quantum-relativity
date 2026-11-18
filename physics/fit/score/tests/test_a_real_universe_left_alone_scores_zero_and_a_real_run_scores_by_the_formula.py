"""On a real universe's own measures: one that is never stepped scores exactly 0, and a stepped one scores hold x (E + S) / 2 between 0 and 1."""
import pytest

from physics.engine import Settings, new_universe, step
from physics.fit.measures.definitions.observer_time import observer_time
from physics.fit.measures.logic.measure_cells import measure_cells
from physics.fit.measures.logic.read_universe import read_universe
from physics.fit.score.logic.score_samples import score_samples
from physics.fit.tests.needs_gpu import needs_gpu

_SETTINGS = Settings(push_strength=40.0, tear_length=200.0)


def _measured(universe):
    return measure_cells(read_universe(universe), _SETTINGS.tear_length)


@needs_gpu
def test_a_real_start_measured_four_times_without_a_step_scores_exactly_zero():
    universe = new_universe(_SETTINGS, 8, 1)
    samples = [{**_measured(universe), "tau": float(index), "gap": None} for index in range(4)]
    scored = score_samples(samples, held_share=samples[0]["lump_share"])
    assert scored["score"] == 0.0 and scored["E"] == 0.0 and scored["S"] == 0.0 and scored["hold"] == 1.0


@needs_gpu
def test_a_real_run_on_a_grid_that_stands_scores_by_the_formula():
    universe = new_universe(_SETTINGS, 8, 1)
    samples = [{**_measured(universe), "tau": 0.0, "gap": None}]
    for _ in range(4):
        for _ in range(6):
            step(universe, _SETTINGS)
        measured = _measured(universe)
        passed = observer_time(6 * _SETTINGS.step_time, samples[-1]["observer_rate"], measured["observer_rate"])
        samples.append({**measured, "tau": samples[-1]["tau"] + passed, "gap": None})
    scored = score_samples(samples)
    assert scored["hold"] == 1.0, "no volume fell in and none is past the tear length"
    assert scored["score"] == pytest.approx(scored["hold"] * (scored["E"] + scored["S"]) / 2.0)
    assert scored["S"] == pytest.approx((scored["Sr"] + scored["Sl"] + scored["Sh"] + scored["Sp"]) / 4.0)
    assert 0.0 < scored["score"] < 1.0 and scored["Sr"] > 0.0, "the energy grows more uneven over regions in this run"
    assert samples[-1]["tau"] < 24 * _SETTINGS.step_time, "packed clocks run slow: less observer time than counter time"
