"""run_universe samples a real universe every so many steps on the observer's clock and scores it; a universe that is over ends early, and one out of wall time is unfinished, both scoring 0 with the reason."""
import pytest

from physics.fit.search.definitions.latin_hypercube import latin_hypercube
from physics.fit.search.definitions.ranges import SEARCHED
from physics.fit.search.definitions.unit_to_overrides import unit_to_overrides
from physics.fit.search.logic.run_universe import run_universe
from physics.fit.tests.needs_gpu import needs_gpu

_HELD = {"push_strength": 40.0, "tear_length": 200.0}


@needs_gpu
def test_a_grid_that_stands_is_sampled_to_the_end_and_scored():
    record = run_universe(_HELD, 1, 8, 24, 4)
    series = record["series"]
    assert series["step"] == [0, 4, 8, 12, 16, 20, 24] and series["time"] == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    assert series["tau"][0] == 0.0 and all(0.0 < later - earlier < 1.0 for earlier, later in zip(series["tau"], series["tau"][1:])), \
        "packed clocks run slow: under one unit of observer time per unit of counter time"
    assert record["steps_done"] == 24 and record["ended"] == "" and not record["unfinished"] and "error" not in record
    assert series["alive"] == [1.0] * 7 and series["spent_share"] == [0.0] * 7 and series["volumes"] == [64] * 7
    assert record["hold"] == 1.0 and record["score"] == pytest.approx((record["E"] + record["S"]) / 2.0)
    assert 0.0 <= record["score"] < 1.0 and set(record["counter"]) == {"score", "E", "dip", "gap_dip"}
    assert record["overrides"] == _HELD and (record["seed"], record["cells"], record["steps"]) == (1, 8, 24)


@needs_gpu
def test_the_default_universe_8_a_side_stands_and_is_sampled_to_the_end():
    record = run_universe({}, 1, 8, 24, 4)
    assert record["ended"] == "" and record["steps_done"] == 24 and record["series"]["volumes"] == [64] * 7


@needs_gpu
def test_a_run_out_of_wall_time_is_unfinished_and_scores_zero():
    record = run_universe(_HELD, 1, 8, 24, 4, seconds=0.0)
    assert record["unfinished"] and record["steps_done"] == 1 and record["score"] == 0.0
    assert record["error"] == "unfinished: 1 of 24 steps in 0.0 s" and record["series"]["step"] == [0]


@needs_gpu
def test_a_universe_whose_collapses_take_all_the_plus_energy_is_over_early_and_scores_zero():
    record = run_universe({"collapse_thin": 1.01}, 1, 8, 40, 4)     # a joint collapses a hundredth above fresh
    assert record["ended"] in ("no energy volume holds any energy", "fewer than 5% of the start's volumes alive")
    assert record["steps_done"] < 40 and record["score"] == 0.0 and record["error"].startswith("ended")


def test_a_name_that_is_no_setting_is_refused_before_anything_runs():
    with pytest.raises(TypeError, match="dark_energy"):
        run_universe({"dark_energy": 1.0}, 1, 8, 24, 4)
    with pytest.raises(ValueError, match="friction"):
        run_universe({"friction": 0.0}, 1, 8, 24, 4)
