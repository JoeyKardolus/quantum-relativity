"""The fit search runs two Latin-hypercube points on a tiny universe, start unevennesses included, records both and summarizes them."""
import json
import math

from physics.fit import fit_search, summarize_search
from physics.fit.fit_search import SEARCHED
from physics.scenes.tests._gpu import needs_gpu


@needs_gpu
def test_the_fit_search_runs_two_points_on_a_tiny_universe(tmp_path):
    results = tmp_path / "results.jsonl"
    records = fit_search(results, cells=5, boxes=4, steps=6, every=1, default_seeds=(), random_points=2,
                         refine_best=0, recheck_top=0)
    assert [record["phase"] for record in records] == ["random", "random"]
    assert len(results.read_text().splitlines()) == 2
    for record in records:
        assert "error" not in record, record.get("trace")
        assert math.isfinite(record["score"]) and math.isfinite(record["E"]) and math.isfinite(record["S"])
        assert record["overrides"]["seal_squeeze"] < record["overrides"]["collapse_squeeze"]
        assert len(record["unit"]) == len(SEARCHED)
        assert 0.0 <= record["overrides"]["unevenness"] <= 0.9
        assert 0.0 <= record["overrides"]["joint_unevenness"] <= 0.9
        assert record["observer_ids"] and len(record["local_time"]) == len(record["time"])
        assert record["local_time"][0] == 0.0
        assert all(later > earlier for earlier, later in zip(record["local_time"], record["local_time"][1:]))
    assert summarize_search([json.loads(line) for line in results.read_text().splitlines()]).startswith("runs 2 errors 0")
