"""Records are appended one JSON line each with their task's phase, id and point, loaded back in order, and a missing file holds none."""
import json

import pytest

from physics.fit.search.logic.load_records import load_records
from physics.fit.search.logic.write_record import write_record


def test_two_records_written_are_two_records_loaded_in_order(tmp_path, capsys):
    results = tmp_path / "results.jsonl"
    assert load_records(results) == []
    first = {"id": "random:0:1", "phase": "random", "unit": [0.25], "seed": 1}
    second = {"id": "hold:3:2", "phase": "hold", "unit": [0.75], "seed": 2}
    write_record(results, first, {"score": 0.125, "seconds": 3.4})
    before = results.read_text()
    write_record(results, second, {"score": 0.0, "error": "the process died without an answer"})
    assert results.read_text().startswith(before) and len(results.read_text().splitlines()) == 2
    assert load_records(results) == [
        {"score": 0.125, "seconds": 3.4, "phase": "random", "id": "random:0:1", "unit": [0.25]},
        {"score": 0.0, "error": "the process died without an answer", "phase": "hold", "id": "hold:3:2", "unit": [0.75]}]
    assert capsys.readouterr().out.splitlines() == ["random:0:1 0.125 3 ", "hold:3:2 0.0 0 the process died without an answer"]


def test_blank_lines_are_skipped_and_a_broken_line_raises(tmp_path):
    results = tmp_path / "results.jsonl"
    results.write_text('{"id": "a"}\n\n{"id": "b"}\n')
    assert [record["id"] for record in load_records(results)] == ["a", "b"]
    results.write_text('{"id": "a"}\n{"id": \n')
    with pytest.raises(json.JSONDecodeError):
        load_records(results)
