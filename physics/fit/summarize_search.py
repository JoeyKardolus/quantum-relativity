"""summarize_search: a plain-text summary of a fit search's records.

Contract: summarize_search(records) -> str. Lists the runs, errors and unfinished runs,
every defaults run, the searched score spread and its parts, how many searched universes
slow then speed up (dip > 0), collapse, tear or seal, the 8 best searched points, each
setting's mean score over seeds (the defaults, and every rechecked point with its
seed-1 search run), and what the fit cannot decide: per searched setting the rank
correlation of its unit-cube place with the score over the random phase, and the
sweep's scores with that one setting moved (a flat row: the score does not read it).
"""
from __future__ import annotations

import collections
import json

import numpy as np

from physics.fit.fit_search import SEARCHED

_KEYS = ["score", "E", "S", "growth", "dip", "Sr", "Sl", "Ss", "Sp"]
_ENDS = ["lumps", "largest_lump", "lump_volumes", "sealed_lumps", "collapses", "splits", "volumes"]


def _brief(record):
    """The scores and end state of one run, rounded."""
    series = record.get("series", {})
    return {key: round(record[key], 3) for key in _KEYS if key in record} | {
        key: series[key][-1] for key in _ENDS if key in series} | {
        "sec": round(record.get("seconds", 0)), "err": record.get("error", "")[:80]}


def _searched_lines(searched):
    """Score spread, what happened in how many universes, and the best points."""
    scores = np.array([record["score"] for record in searched])
    lines = [f"searched {scores.size} score quantiles 0/10/50/90/100: {np.round(np.quantile(scores, [0, .1, .5, .9, 1]), 3)}"]
    for key in ["E", "S", "Sr", "Sl", "Ss", "Sp"]:
        values = np.array([record[key] for record in searched if key in record])
        if values.size:
            lines.append(f"  {key} quantiles {np.round(np.quantile(values, [0, .1, .5, .9, 1]), 3)}")
    ended = [{key: record["series"][key][-1] for key in _ENDS} for record in searched if "series" in record]
    lines.append(f"slow then speed up (dip > 0): {sum(record.get('dip', -1) > 0 and record.get('E', 0) > 0 for record in searched)}; "
                 f"collapsed: {sum(end['collapses'] > 0 for end in ended)}; tore: {sum(end['splits'] > 0 for end in ended)}; "
                 f"sealed a lump: {sum(end['sealed_lumps'] > 0 for end in ended)}; "
                 f"scored 0: {int((scores == 0).sum())}")
    for record in sorted(searched, key=lambda record: -record["score"])[:8]:
        overrides = {key: float(f"{value:.4g}") for key, value in record["overrides"].items()}
        lines.append(f"top {record['id']} {overrides} {_brief(record)}")
    return lines


def _group_lines(records):
    """The defaults and each rechecked setting: its seeds, scores, mean and spread."""
    groups = collections.defaultdict(list)
    rechecked = {json.dumps(record["overrides"], sort_keys=True) for record in records if record["phase"] == "recheck"}
    for record in records:
        name = json.dumps(record["overrides"], sort_keys=True)
        if record["phase"] in ("recheck", "defaults") or (name in rechecked and record["phase"] != "sweep"):
            groups[name].append(record)
    lines = []
    for name, group in groups.items():
        values = [record["score"] for record in group]
        lines.append(f"group {name} seeds {[record['seed'] for record in group]} scores {np.round(values, 3)} "
                     f"mean {round(float(np.mean(values)), 3)} sd {round(float(np.std(values)), 3)}")
    return lines


def _ranks(values):
    """Each value's rank, ties sharing their mean rank."""
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="stable")
    ranks = np.empty(values.size)
    ranks[order] = np.arange(values.size)
    for value in np.unique(values):
        ranks[values == value] = ranks[values == value].mean()
    return ranks


def _flat_lines(records):
    """Per searched setting: rank correlation with the score over the random phase, and the sweep's scores."""
    lines = []
    random = [record for record in records if record["phase"] == "random"]
    sweep = [record for record in records if record["phase"] == "sweep"]
    for axis, name in enumerate(SEARCHED):
        line = f"setting {name}:"
        if len(random) >= 8:
            places = _ranks([record["unit"][axis] for record in random])
            scores = _ranks([record["score"] for record in random])
            if places.std() > 0 and scores.std() > 0:
                line += f" random-phase rank correlation {np.corrcoef(places, scores)[0, 1]:+.2f};"
        moved = sorted((record["unit"][axis], record["score"]) for record in sweep if record["id"].startswith(f"sweep:{name}@"))
        if moved:
            scores = [score for _, score in moved]
            line += f" sweep {[(round(place, 2), round(score, 3)) for place, score in moved]} range {max(scores) - min(scores):.3f}"
        lines.append(line)
    return lines


def summarize_search(records):
    """The summary text of a list of fit_search records."""
    lines = [f"runs {len(records)} errors {sum('error' in record for record in records)} "
             f"unfinished {sum(bool(record.get('unfinished')) for record in records)}"]
    lines += [f"default seed {record['seed']} {_brief(record)}" for record in records if record["phase"] == "defaults"]
    searched = [record for record in records if record["phase"] == "random" or record["phase"].startswith("refine")]
    if searched:
        lines += _searched_lines(searched)
    lines += _group_lines(records)
    lines += _flat_lines(records)
    return "\n".join(lines) + "\n"
