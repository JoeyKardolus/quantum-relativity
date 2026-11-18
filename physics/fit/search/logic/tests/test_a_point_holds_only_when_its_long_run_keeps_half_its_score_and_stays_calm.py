"""A searched point holds only when its four-times-longer run keeps at least half its short score and stays calm; holders come first."""
from physics.fit.search.logic.find_holders import find_holders


def _record(phase, unit, seed, score, hold=1.0, ended=""):
    return {"phase": phase, "unit": unit, "seed": seed, "score": score, "hold": hold, "ended": ended,
            "overrides": {"pull_strength": unit[0]}}


def test_a_runaway_caught_midway_does_not_hold_and_a_steady_point_does():
    steady, runaway, wilted, torn = [0.1], [0.2], [0.3], [0.4]
    records = [_record("random", steady, 1, 0.10), _record("random", runaway, 1, 0.30),
               _record("refine1", wilted, 1, 0.20), _record("random", torn, 1, 0.20),
               _record("hold", steady, 1, 0.12), _record("hold", steady, 2, 0.08), _record("hold", steady, 3, 0.10),
               _record("hold", runaway, 1, 0.01, hold=0.04, ended="more than two tears per start volume"),
               _record("hold", wilted, 1, 0.09), _record("hold", torn, 1, 0.15, hold=0.6)]
    points = find_holders(records)
    assert [point["unit"] for point in points] == [steady, torn, wilted, runaway], "the holder first, the rest by mean"
    assert [point["holds"] for point in points] == [True, False, False, False]
    assert points[0]["long"] == {1: 0.12, 2: 0.08, 3: 0.10} and abs(points[0]["mean"] - 0.10) < 1e-12
    assert points[3]["ended"] == "more than two tears per start volume" and points[3]["short"] == 0.30


def test_a_point_that_scored_nothing_never_holds_and_no_hold_phase_gives_no_points():
    nothing = [0.5]
    assert find_holders([_record("random", nothing, 1, 0.0)]) == []
    points = find_holders([_record("random", nothing, 1, 0.0), _record("hold", nothing, 1, 0.0)])
    assert len(points) == 1 and not points[0]["holds"]


def test_a_point_run_long_only_on_another_seed_does_not_hold():
    point = [0.7]
    points = find_holders([_record("random", point, 1, 0.2), _record("hold", point, 2, 0.2)])
    assert points[0]["holds"] is False and points[0]["long"] == {2: 0.2} and points[0]["ended"] == ""
