"""The summary of hand-made records counts runs, errors and ends, names the holder, and shows a flat sweep for a setting the score does not read."""
from physics.fit.search.definitions.ranges import SEARCHED
from physics.fit.search.logic.summarize_search import summarize_search


def _unit(place):
    return [place] + [0.5] * (len(SEARCHED) - 1)


def _searched(index, place, score, hold=1.0, ended="", tears=0):
    return {"id": f"random:{index}:1", "phase": "random", "seed": 1, "unit": _unit(place), "score": score, "hold": hold,
            "E": score, "S": score, "growth": score, "dip": 0.2 if score else -0.1, "Sl": score, "ended": ended,
            "overrides": {"pull_strength": place}, "seconds": 2.0,
            "series": {"lumps": [0, 3], "largest_lump": [0, 2], "lump_volumes": [0, 4], "collapses": [0, 0],
                       "merges": [0, 0], "tears": [0, tears], "volumes": [64, 64], "amount": [100, 100]}}


def _records():
    records = [{"id": "defaults:0:1", "phase": "defaults", "seed": 1, "unit": None, "score": 0.0, "overrides": {}}]
    records += [_searched(index, index / 8.0, index / 10.0) for index in range(8)]
    records.append(_searched(8, 0.95, 0.0, hold=0.02, ended="more than two tears per start volume", tears=200))
    records.append({"id": "random:9:1", "phase": "random", "seed": 1, "unit": _unit(0.99), "score": 0.0,
                    "overrides": {}, "error": "the process died without an answer"})
    records.append({"id": "hold:0:1", "phase": "hold", "seed": 1, "unit": _unit(7 / 8.0), "score": 0.6, "hold": 0.95,
                    "ended": "", "overrides": {}})
    for place, score in ((0.1, 0.3), (0.9, 0.3)):
        unit = _unit(7 / 8.0)
        unit[1] = place
        records.append({"id": f"sweep:light_speed@{place}:1", "phase": "sweep", "seed": 1, "unit": unit,
                        "score": score, "overrides": {}})
    return records


def test_the_summary_of_hand_made_records():
    lines = summarize_search(_records()).splitlines()
    assert lines[0] == "runs 14 errors 1 unfinished 0"
    assert lines[1].startswith("default seed 1 {'score': 0.0")
    assert lines[2].startswith("searched 10 score quantiles 0/10/50/90/100: [0.")
    happened = next(line for line in lines if line.startswith("slow then speed up"))
    assert happened == "slow then speed up (dip > 0): 7; collapsed: 0; tore: 1; scored 0: 3"
    ended = next(line for line in lines if line.startswith("how they ended"))
    assert ended == "how they ended: {'ran to the end': 9, 'more than two tears per start volume': 1}"
    assert "held together (hold above 0.9): 8; of those grew: 7, gained lump share: 7, both: 7" in lines
    assert next(line for line in lines if line.startswith("top ")).startswith("top random:7:1 {'pull_strength': 0.875}")
    assert "defaults mean over 1 seeds 0.0" in lines
    assert any(line.startswith("hold phase: 1 points run long, 1 held") for line in lines)
    assert any(line.startswith("holder short 0.700 long {1: 0.6} mean 0.600") for line in lines)


def test_every_searched_setting_has_its_line_and_a_flat_sweep_shows_a_range_of_zero():
    lines = [line for line in summarize_search(_records()).splitlines() if line.startswith("setting ")]
    assert [line.split()[1].rstrip(":") for line in lines] == SEARCHED
    assert lines[0] == "setting pull_strength: random-phase rank correlation +0.13;", \
        "ranks 0 .. 9 against 1, 3 .. 9, 1, 1: the score rose with the setting but the top two scored 0"
    assert lines[1].endswith("sweep (mean over seeds) [(0.1, 0.3), (0.9, 0.3)] range 0.000")
    assert lines[2] == "setting push_strength:", "ten equal places and never swept: nothing to say"


def test_no_records_give_only_the_count_and_the_settings():
    lines = summarize_search([]).splitlines()
    assert lines[0] == "runs 0 errors 0 unfinished 0" and len(lines) == 1 + len(SEARCHED)
