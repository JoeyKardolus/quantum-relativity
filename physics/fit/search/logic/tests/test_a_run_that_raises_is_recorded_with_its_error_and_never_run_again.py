"""run_tasks records a run that raises in its own process with score 0 and its error, skips ids already in the file, and starts nothing after its time is up."""
import time

from physics.fit.search.logic.load_records import load_records
from physics.fit.search.logic.run_tasks import run_tasks

_SIZE = (8, 4, 2)


def _bad_task(index):
    return {"id": f"random:{index}:1", "phase": "random", "seed": 1, "unit": [0.5], "overrides": {"dark_energy": 1.0}}


def test_two_runs_that_raise_are_two_records_with_the_error_and_a_second_call_runs_nothing(tmp_path):
    results = tmp_path / "results.jsonl"
    assert run_tasks(results, [_bad_task(0), _bad_task(1)], _SIZE, 2, 60.0, None, "cpu") is True
    records = load_records(results)
    assert sorted(record["id"] for record in records) == ["random:0:1", "random:1:1"]
    for record in records:
        assert record["score"] == 0.0 and "dark_energy" in record["error"] and "TypeError" in record["trace"]
        assert record["phase"] == "random" and record["unit"] == [0.5]
    before = results.read_text()
    assert run_tasks(results, [_bad_task(0), _bad_task(1)], _SIZE, 2, 60.0, None, "cpu") is True
    assert results.read_text() == before, "both ids are in the file: nothing is run again"


def test_after_its_time_is_up_no_new_run_is_started(tmp_path):
    results = tmp_path / "results.jsonl"
    assert run_tasks(results, [_bad_task(0)], _SIZE, 2, 60.0, time.time() - 1.0, "cpu") is False
    assert load_records(results) == []
    assert run_tasks(results, [], _SIZE, 2, 60.0, time.time() - 1.0, "cpu") is True, "nothing was waiting, so nothing was cut"
