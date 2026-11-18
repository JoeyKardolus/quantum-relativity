"""Summarize search: a plain-text summary of a fit search's records.

Contract: `summarize_search(records)` returns a str, one fact a line:
  the runs, errors and unfinished runs; every defaults run;
  the searched score spread and its parts; how many searched universes slow then speed
  up (dip > 0), collapse or tear, how they ended and how many held together; the 8 best
  searched points; the defaults' mean over seeds;
  the hold phase: how many points were run long, how many held, the 12 best with their
  short and long scores;
  what the fit cannot decide: per searched setting the rank correlation of its unit-cube
  place with the score over the random phase (with 8 runs or more), and the sweep's
  scores, the mean over its seeds, with that one setting moved (a flat row: the score
  does not read it).
"""
from __future__ import annotations

import collections

import numpy as np

from physics.fit.search.definitions.point_holds import CALM
from physics.fit.search.definitions.ranges import SEARCHED
from physics.fit.search.definitions.searched_phase import searched_phase
from physics.fit.search.logic.find_holders import find_holders

_KEYS = ["score", "E", "S", "hold", "E_size", "E_gap", "growth", "dip", "gap_growth", "Sr", "Sl", "Sh", "Sp"]
_PARTS = ["E", "S", "hold", "E_size", "E_gap", "Sr", "Sl", "Sh", "Sp"]
_ENDS = ["lumps", "largest_lump", "lump_volumes", "collapses", "merges", "tears", "volumes", "amount"]
_QUANTILES = [0, .1, .5, .9, 1]


def _brief(record: dict) -> dict:
    """The scores and end state of one run, rounded."""
    series = record.get("series", {})
    return {key: round(record[key], 3) for key in _KEYS if key in record} | {
        key: series[key][-1] for key in _ENDS if key in series} | {
        "sec": round(record.get("seconds", 0)), "err": record.get("error", "")[:80]}


def _short(overrides: dict) -> dict:
    """Overrides rounded to four figures."""
    return {key: float(f"{value:.4g}") for key, value in overrides.items()}


def _spread_lines(searched: list) -> list:
    """The searched scores' quantiles and those of every part."""
    scores = np.array([record["score"] for record in searched])
    lines = [f"searched {scores.size} score quantiles 0/10/50/90/100: {np.round(np.quantile(scores, _QUANTILES), 3)}"]
    for key in _PARTS:
        values = np.array([record[key] for record in searched if key in record])
        if values.size:
            lines.append(f"  {key} quantiles {np.round(np.quantile(values, _QUANTILES), 3)}")
    return lines


def _happened_lines(searched: list) -> list:
    """What happened in how many searched universes, how they ended and how many held together."""
    ended = [{key: record["series"][key][-1] for key in _ENDS} for record in searched if "series" in record]
    dipped = sum(record.get("dip", -1) > 0 and record.get("E", 0) > 0 for record in searched)
    over = collections.Counter(record.get("ended") or "ran to the end" for record in searched)
    calm = [record for record in searched if record.get("hold", 0.0) > CALM]
    return [f"slow then speed up (dip > 0): {dipped}; collapsed: {sum(end['collapses'] > 0 for end in ended)}; "
            f"tore: {sum(end['tears'] > 0 for end in ended)}; "
            f"scored 0: {sum(record['score'] == 0 for record in searched)}",
            f"how they ended: {dict(over)}",
            f"held together (hold above {CALM}): {len(calm)}; of those grew: "
            f"{sum(record['growth'] > 0 for record in calm)}, gained lump share: "
            f"{sum(record['Sl'] > 0 for record in calm)}, both: "
            f"{sum(record['growth'] > 0 and record['Sl'] > 0 for record in calm)}"]


def _top_lines(searched: list) -> list:
    """The 8 best searched points."""
    best = sorted(searched, key=lambda record: -record["score"])[:8]
    return [f"top {record['id']} {_short(record['overrides'])} {_brief(record)}" for record in best]


def _hold_lines(records: list) -> list:
    """The defaults' mean over seeds, and the hold phase's points: short score, long scores, whether each held."""
    lines = []
    defaults = [record["score"] for record in records if record["phase"] == "defaults"]
    if defaults:
        lines.append(f"defaults mean over {len(defaults)} seeds {round(float(np.mean(defaults)), 4)}")
    points = find_holders(records)
    if points:
        over = collections.Counter(point["ended"] or "ran to the end" for point in points)
        lines.append(f"hold phase: {len(points)} points run long, {sum(point['holds'] for point in points)} held; "
                     f"their long seed-1 runs: {dict(over)}")
    for point in points[:12]:
        long = {seed: round(score, 3) for seed, score in point["long"].items()}
        lines.append(f"{'holder' if point['holds'] else 'fell'} short {point['short']:.3f} long {long} "
                     f"mean {point['mean']:.3f} {point['ended']} {_short(point['overrides'])}")
    return lines


def _ranks(values) -> np.ndarray:
    """Each value's rank, ties sharing their mean rank."""
    values = np.asarray(values, dtype=np.float64)
    ranks = np.empty(values.size)
    ranks[np.argsort(values, kind="stable")] = np.arange(values.size)
    for value in np.unique(values):
        ranks[values == value] = ranks[values == value].mean()
    return ranks


def _correlation(random: list, axis: int) -> str:
    """The rank correlation of one setting's place with the score over the random phase, as text ("" when too few)."""
    if len(random) < 8:
        return ""
    places = _ranks([record["unit"][axis] for record in random])
    scores = _ranks([record["score"] for record in random])
    if not (places.std() > 0 and scores.std() > 0):
        return ""
    return f" random-phase rank correlation {np.corrcoef(places, scores)[0, 1]:+.2f};"


def _swept(sweep: list, axis: int, name: str) -> str:
    """The sweep's mean scores with one setting moved, as text ("" when it was not swept)."""
    places = collections.defaultdict(list)
    for record in sweep:
        if record["id"].startswith(f"sweep:{name}@"):
            places[record["unit"][axis]].append(record["score"])
    moved = sorted((place, float(np.mean(scores))) for place, scores in places.items())
    if not moved:
        return ""
    scores = [score for _, score in moved]
    return (f" sweep (mean over seeds) {[(round(place, 2), round(score, 3)) for place, score in moved]} "
            f"range {max(scores) - min(scores):.3f}")


def _setting_lines(records: list) -> list:
    """Per searched setting: its rank correlation with the score and its sweep."""
    random = [record for record in records if record["phase"] == "random"]
    sweep = [record for record in records if record["phase"] == "sweep"]
    return [f"setting {name}:" + _correlation(random, axis) + _swept(sweep, axis, name)
            for axis, name in enumerate(SEARCHED)]


def summarize_search(records: list) -> str:
    """The summary text of a list of search records."""
    lines = [f"runs {len(records)} errors {sum('error' in record for record in records)} "
             f"unfinished {sum(bool(record.get('unfinished')) for record in records)}"]
    lines += [f"default seed {record['seed']} {_brief(record)}" for record in records if record["phase"] == "defaults"]
    searched = [record for record in records if searched_phase(record["phase"])]
    if searched:
        lines += _spread_lines(searched) + _happened_lines(searched) + _top_lines(searched)
    lines += _hold_lines(records) + _setting_lines(records)
    return "\n".join(lines) + "\n"
