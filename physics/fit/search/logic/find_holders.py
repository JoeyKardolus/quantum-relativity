"""Find holders: which searched points still stand when run four times longer.

Contract: `find_holders(records)` returns a list of dicts, best first, one per searched
point the hold phase reran (the best calm points of the short runs, run again for longer
on seed 1, the holders then on more seeds):
  unit, overrides   the point
  short             its score in the short search run (seed 1)
  long              its long runs' scores by seed, {seed: score}
  ended             what ended its long seed-1 run early ("" when it ran to the end)
  holds             `point_holds` of the short score and the long seed-1 run; False
                    when that run is missing
  mean              the mean of its long scores over the seeds run
Holders come first, by mean; then the others, by mean. The fitted point is the first
when it holds; with no holder nothing is fitted.
"""
from __future__ import annotations

import json

from physics.fit.search.definitions.point_holds import point_holds
from physics.fit.search.definitions.searched_phase import searched_phase


def _long_runs(records: list) -> dict:
    """The hold phase's records by point and then by seed."""
    points: dict = {}
    for record in records:
        if record["phase"] == "hold":
            points.setdefault(json.dumps(record["unit"]), {})[record["seed"]] = record
    return points


def _point(source: dict, runs: dict) -> dict:
    """One reran point: its short score, its long scores and whether it holds."""
    first = runs.get(1)
    scores = {seed: run["score"] for seed, run in sorted(runs.items())}
    return {"unit": source["unit"], "overrides": source["overrides"], "short": source["score"], "long": scores,
            "ended": first.get("ended", "") if first else "",
            "holds": bool(first) and point_holds(source["score"], first["score"], first.get("hold", 0.0)),
            "mean": sum(scores.values()) / len(scores)}


def find_holders(records: list) -> list:
    """Every point the hold phase reran, with whether it held, holders first."""
    short = {json.dumps(record["unit"]): record for record in records if searched_phase(record["phase"])}
    found = [_point(short[name], runs) for name, runs in _long_runs(records).items()]
    return sorted(found, key=lambda point: (not point["holds"], -point["mean"]))
