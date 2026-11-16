"""summarize_search: a plain-text summary of a fit search's records.

Contract: summarize_search(records) -> str. Lists the runs and errors, every defaults run, the
searched score spread and its parts, how many runs slow then speed up (dip > 0), the 8 best
searched points, and each rechecked setting's scores over seeds with mean and spread.
"""
from __future__ import annotations

import collections
import json

import numpy as np

_KEYS = ["score", "E", "S", "dip", "Sc", "Sl", "St"]


def _brief(record):
    """The scores and end state of one run, rounded."""
    return {key: round(record[key], 3) for key in _KEYS if key in record} | {
        "grow": round(record.get("occupied_growth_end", float("nan")), 3),
        "edge": round(record.get("edge_share_end", float("nan")), 4),
        "lumps_end": (record.get("lumps") or [None])[-1], "sec": round(record.get("seconds", 0), 1),
        "err": record.get("error", "")[:80]}


def _searched_lines(searched):
    """Score spread, dip count and the best points of the random and refine phases."""
    scores = np.array([record["score"] for record in searched])
    lines = [f"searched {scores.size} score quantiles 0/10/50/90/100: {np.round(np.quantile(scores, [0, .1, .5, .9, 1]), 3)}"]
    for key in ["E", "S", "Sc", "Sl", "St"]:
        values = np.array([record[key] for record in searched if key in record])
        if values.size:
            lines.append(f"  {key} quantiles {np.round(np.quantile(values, [0, .1, .5, .9, 1]), 3)}")
    lines.append(f"positive dip (slow then speed up): {sum(record.get('dip', -1) > 0 for record in searched)}, "
                 f"lumps at end > 0: {sum((record.get('lumps') or [0])[-1] > 0 for record in searched)}")
    for record in sorted(searched, key=lambda record: -record["score"])[:8]:
        overrides = {key: round(value, 4) for key, value in record["overrides"].items()}
        lines.append(f"top {record['phase']} {overrides} {_brief(record)}")
    return lines


def _group_lines(records):
    """Each defaults or rechecked setting: its seeds, scores, mean and spread."""
    groups = collections.defaultdict(list)
    for record in records:
        if record["phase"] in ("recheck", "defaults"):
            groups[json.dumps(record["overrides"], sort_keys=True)].append(record)
    lines = []
    for name, group in groups.items():
        values = [record["score"] for record in group]
        lines.append(f"group {name[:160]} seeds {[record['seed'] for record in group]} scores {np.round(values, 3)} "
                     f"mean {round(float(np.mean(values)), 3)} sd {round(float(np.std(values)), 3)}")
    return lines


def summarize_search(records):
    """The summary text of a list of fit_search records."""
    lines = [f"runs {len(records)} errors {sum('error' in record for record in records)}"]
    lines += [f"default seed {record['seed']} {_brief(record)}" for record in records if record["phase"] == "defaults"]
    searched = [record for record in records if record["phase"] in ("random", "refine")]
    if searched:
        lines += _searched_lines(searched)
    lines += _group_lines(records)
    return "\n".join(lines) + "\n"
