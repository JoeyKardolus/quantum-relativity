"""The whole search on tiny wrapping universes: two random points side by side, the hold phase and the sweep, every run recorded once, and a stopped search resumed."""
import math

from physics.fit.search.definitions.ranges import LINEAR_RANGES, LOG_RANGES, SEARCHED
from physics.fit.search.definitions.search_plan import SearchPlan
from physics.fit.search.logic.run_search import run_search
from physics.fit.search.logic.summarize_search import summarize_search
from physics.fit.tests.needs_gpu import needs_gpu

_TINY = dict(cells=8, steps=8, every=2, workers=2, default_seeds=(), random_points=2, refine_rounds=0, hold_top=0,
             sweep_at=())


@needs_gpu
def test_two_random_points_are_run_recorded_and_summarized_and_a_second_call_runs_nothing(tmp_path):
    results = tmp_path / "out" / "results.jsonl"
    records = run_search(results, SearchPlan(**_TINY))
    assert sorted(record["id"] for record in records) == ["random:0:1", "random:1:1"]
    assert len(results.read_text().splitlines()) == 2
    for record in records:
        assert "trace" not in record, record.get("trace")
        assert record.get("error", "ended").startswith("ended"), "the only error is a universe that was over early"
        assert 0.0 <= record["score"] <= 1.0 and 0.0 <= record["E"] <= 1.0 and 0.0 <= record["S"] <= 1.0
        assert len(record["unit"]) == len(SEARCHED) == 10
        for name, (low, high) in (LOG_RANGES | LINEAR_RANGES).items():
            assert low <= record["overrides"][name] <= high
        series = record["series"]
        assert series["step"] == [0, 2, 4, 6, 8][:len(series["step"])] and series["tau"][0] == 0.0
        assert record["ended"] or len(series["step"]) == 5
        assert all(later > earlier for earlier, later in zip(series["tau"], series["tau"][1:]))
        assert all(math.isfinite(value) and value > 0 for value in series["size"])
    assert run_search(results, SearchPlan(**_TINY)) == records
    assert summarize_search(records).startswith("runs 2 errors ")


@needs_gpu
def test_the_hold_phase_runs_the_calm_points_again_for_longer_and_the_sweep_moves_one_setting(tmp_path):
    plan = SearchPlan(**{**_TINY, "refine_rounds": 1, "refine_best": 1, "refine_count": 1, "hold_top": 2,
                         "hold_steps": 16, "hold_seeds": (2,), "sweep_at": (0.5,), "sweep_seeds": (1,)})
    records = run_search(tmp_path / "results.jsonl", plan)
    assert len({record["id"] for record in records}) == len(records), "no run is recorded twice"
    searched = {tuple(record["unit"]): record for record in records if record["phase"] in ("random", "refine1")}
    held = [record for record in records if record["phase"] == "hold"]
    swept = [record for record in records if record["phase"] == "sweep"]
    assert len(searched) == 3 and len(swept) == len(SEARCHED)
    assert sorted(record["id"] for record in swept) == sorted(f"sweep:{name}@0.5:1" for name in SEARCHED)
    assert len(held) <= 4 and all(tuple(record["unit"]) in searched and record["steps"] == 16 for record in held)
    assert all(searched[tuple(record["unit"])]["hold"] > 0.9 for record in held), "only calm points are run long"
    assert "hold phase" in summarize_search(records) or not held


@needs_gpu
def test_a_search_out_of_minutes_starts_no_new_run_and_resumes_where_it_stood(tmp_path):
    results = tmp_path / "results.jsonl"
    assert run_search(results, SearchPlan(**_TINY), minutes=0.0) == []
    assert sorted(record["id"] for record in run_search(results, SearchPlan(**_TINY))) == ["random:0:1", "random:1:1"]
