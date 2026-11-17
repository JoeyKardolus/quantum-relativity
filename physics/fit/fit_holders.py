"""fit_holders: which searched points still stand when run four times longer.

Contract: fit_holders(records) -> list of dicts, best first, one per searched point
that the search's hold phase reran (fit_search: the best points of the short runs, run
again for hold_steps on seed 1, the holders then on more seeds):
  unit, overrides   the point
  short             its score in the short search run (seed 1)
  long              its long runs' scores by seed, {seed: score}
  ended             what ended its long seed-1 run early ("" when it ran to the end)
  holds             True when the long seed-1 score is at least HALF_KEPT (0.5) of the
                    short one and the long run's hold factor stayed above CALM (0.9):
                    a universe whose short score was no runaway caught midway
  mean              the mean of its long scores over the seeds run
Holders come first, by mean; then the others, by mean. The fitted point is the first
when it holds (fit_search sweeps round it); with no holder nothing is fitted.
"""
from __future__ import annotations

import json

HALF_KEPT = 0.5
CALM = 0.9


def fit_holders(records: list) -> list:
    """Every point the hold phase reran, with whether it held, holders first."""
    short = {json.dumps(record["unit"]): record for record in records
             if record["phase"] == "random" or record["phase"].startswith("refine")}
    points = {}
    for record in records:
        if record["phase"] == "hold":
            points.setdefault(json.dumps(record["unit"]), {})[record["seed"]] = record
    found = []
    for name, runs in points.items():
        first, source = runs.get(1), short[name]
        scores = {seed: run["score"] for seed, run in sorted(runs.items())}
        found.append({"unit": source["unit"], "overrides": source["overrides"], "short": source["score"],
                      "long": scores, "ended": first.get("ended", "") if first else "",
                      "holds": bool(first and first["score"] >= HALF_KEPT * source["score"] > 0.0
                                    and first.get("hold", 0.0) > CALM),
                      "mean": sum(scores.values()) / len(scores)})
    return sorted(found, key=lambda point: (not point["holds"], -point["mean"]))
