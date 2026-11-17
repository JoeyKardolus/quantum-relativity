"""The fit search runs two Latin-hypercube points side by side on a tiny wrapping universe, records both and summarizes them."""
import json
import math

from physics.fit import fit_search, summarize_search
from physics.fit.fit_search import LINEAR_RANGES, LOG_RANGES, SEARCHED, START_RANGES
from physics.scenes.tests._gpu import needs_gpu


@needs_gpu
def test_the_fit_search_runs_two_points_on_a_tiny_universe(tmp_path):
    results = tmp_path / "results.jsonl"
    records = fit_search(results, cells=8, boxes=6, steps=8, every=2, workers=2, default_seeds=(), random_points=2,
                         refine_rounds=0, recheck_top=0, sweep_at=())
    assert sorted(record["id"] for record in records) == ["random:0:1", "random:1:1"]
    assert len(results.read_text().splitlines()) == 2
    for record in records:
        assert "error" not in record, record.get("trace")
        assert 0.0 <= record["score"] <= 1.0 and 0.0 <= record["E"] <= 1.0 and 0.0 <= record["S"] <= 1.0
        assert len(record["unit"]) == len(SEARCHED) == 10
        for name, (low, high) in (LOG_RANGES | LINEAR_RANGES | START_RANGES).items():
            assert low <= record["overrides"][name] <= high
        series = record["series"]
        assert series["step"] == [0, 2, 4, 6, 8] and series["tau"][0] == 0.0
        assert all(later > earlier for earlier, later in zip(series["tau"], series["tau"][1:]))
        assert all(math.isfinite(value) and value > 0 for value in series["room"])
    assert fit_search(results, cells=8, boxes=6, steps=8, every=2, workers=2, default_seeds=(), random_points=2,
                      refine_rounds=0, recheck_top=0, sweep_at=()) == records
    assert summarize_search([json.loads(line) for line in results.read_text().splitlines()]).startswith("runs 2 errors 0")
