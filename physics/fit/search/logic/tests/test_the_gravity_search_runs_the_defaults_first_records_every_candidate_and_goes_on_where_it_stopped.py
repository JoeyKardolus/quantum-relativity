"""`gravity_search` with a scene that returns made-up numbers: what it runs, records, skips and returns."""
import json

from physics.fit.search.logic.gravity_search import gravity_search

GOOD = {"stand_steps": 10, "stand_collapses": 0, "stand_tears": 0, "fall_closed": 0.3, "fall_kept": 0.9,
        "fall_sped_up": 1.0, "lone_drift": 0.0, "coast_travelled": 5.0, "coast_early_speed": 0.01,
        "coast_late_speed": 0.01, "coast_kept": 0.9, "pull_power": -2.0, "law_power": -2.0, "orbit_turns": 1.0,
        "orbit_kept": 0.9, "orbit_big_kept": 0.9, "fall_apart": [4.0, 3.0]}


def test_the_defaults_run_first_every_candidate_is_recorded_and_a_second_call_runs_nothing_again(tmp_path):
    calls = []

    def scene(cells, steps, every, settings):
        calls.append(settings)
        if len(calls) == 2:
            raise RuntimeError("this candidate cannot run")
        return dict(GOOD, fall_closed=0.3 if len(calls) == 3 else 0.1)

    out = tmp_path / "phase1" / "gravity.jsonl"
    records = gravity_search(out, 3, seed=5, cells=24, steps=10, every=5, scene=scene)
    assert len(calls) == 4 and calls[0].pull_strength == 1e-7, "the defaults, then three candidates"
    lines = [json.loads(line) for line in out.read_text().splitlines()]
    assert [line["id"] for line in lines] == [0, 1, 2, 3] and lines[0]["overrides"] == {}
    assert "error" in lines[1] and lines[1]["parts"]["score"] == 0.0
    assert "fall_apart" not in lines[2]["numbers"], "lists are left out of the record"
    assert records[0]["id"] == 2 and records[0]["parts"]["score"] > records[1]["parts"]["score"] > 0.0
    again = gravity_search(out, 3, seed=5, cells=24, steps=10, every=5, scene=scene)
    assert len(calls) == 4 and [record["id"] for record in again] == [record["id"] for record in records]
