"""The fit search runs two Latin-hypercube points side by side on a tiny wrapping universe, records both and summarizes them."""
import json
import math

from physics.fit import fit_search, summarize_search
from physics.fit.fit_search import LINEAR_RANGES, LOG_RANGES, SEARCHED
from physics.scenes.tests._gpu import needs_gpu


@needs_gpu
def test_the_fit_search_runs_two_points_on_a_tiny_universe(tmp_path):
    results = tmp_path / "results.jsonl"
    records = fit_search(results, cells=8, steps=8, every=2, workers=2, default_seeds=(), random_points=2,
                         refine_rounds=0, hold_top=0, sweep_at=())
    assert sorted(record["id"] for record in records) == ["random:0:1", "random:1:1"]
    assert len(results.read_text().splitlines()) == 2
    for record in records:
        assert "trace" not in record, record.get("trace")
        assert record.get("error", "ended").startswith("ended"), "the only error is a universe that was over early"
        assert 0.0 <= record["score"] <= 1.0 and 0.0 <= record["E"] <= 1.0 and 0.0 <= record["S"] <= 1.0
        assert len(record["unit"]) == len(SEARCHED) == 15
        for name, (low, high) in (LOG_RANGES | LINEAR_RANGES).items():
            assert low <= record["overrides"][name] <= high
        series = record["series"]
        assert series["step"] == [0, 2, 4, 6, 8][:len(series["step"])] and series["tau"][0] == 0.0
        assert record["ended"] or len(series["step"]) == 5
        assert all(later > earlier for earlier, later in zip(series["tau"], series["tau"][1:]))
        assert all(math.isfinite(value) and value > 0 for value in series["size"])
    assert fit_search(results, cells=8, steps=8, every=2, workers=2, default_seeds=(), random_points=2,
                      refine_rounds=0, hold_top=0, sweep_at=()) == records
    assert summarize_search([json.loads(line) for line in results.read_text().splitlines()]).startswith("runs 2 errors ")


@needs_gpu
def test_the_hold_phase_runs_the_calm_points_again_for_longer_and_the_sweep_moves_one_setting(tmp_path):
    records = fit_search(tmp_path / "results.jsonl", cells=8, steps=8, every=2, workers=2, default_seeds=(),
                         random_points=2, refine_rounds=1, refine_best=1, refine_count=1, hold_top=2, hold_steps=16,
                         hold_seeds=(2,), sweep_at=(0.5,), sweep_seeds=(1,))
    searched = {tuple(record["unit"]): record for record in records if record["phase"] in ("random", "refine1")}
    held = [record for record in records if record["phase"] == "hold"]
    swept = [record for record in records if record["phase"] == "sweep"]
    assert len(searched) == 3 and len(swept) == len(SEARCHED)
    assert len(held) <= 4 and all(tuple(record["unit"]) in searched and record["steps"] == 16 for record in held)
    assert all(searched[tuple(record["unit"])]["hold"] > 0.9 for record in held), "only calm points are run long"
    assert "hold phase" in summarize_search(records) or not held


@needs_gpu
def test_a_search_out_of_minutes_starts_no_new_run_and_resumes_where_it_stood(tmp_path):
    tiny = dict(cells=8, steps=8, every=2, workers=2, default_seeds=(), random_points=2, refine_rounds=0, hold_top=0,
                sweep_at=())
    assert fit_search(tmp_path / "results.jsonl", minutes=0.0, **tiny) == []
    assert sorted(record["id"] for record in fit_search(tmp_path / "results.jsonl", **tiny)) == ["random:0:1", "random:1:1"]
