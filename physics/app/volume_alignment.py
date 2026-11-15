"""Which volume after a topology change was which volume before it, for the drawing only.

Contract: `volume_alignment(old_kind, old_energy, new_kind, new_energy)` returns an int64
array with one entry per new volume: the index of the old volume it continues, or -1 for a
volume that is new (a split's second half or its new joint). The State carries no ids, so
this aligns the two volume lists in order (an edit distance): a kept volume must keep its
kind and its place in the order, deleting an old volume or adding a new one costs 1, and
continuing one costs its relative energy change over TOLERANCE, capped at JUMP_COST (so a
volume whose energy jumped, the survivor of a joint collapse or the first half of a split,
still continues itself rather than being deleted and re-added). It assumes the engine
keeps the order of the volumes it keeps (deletes and appends); if it reorders them the
drawing scrambles for a few frames until the layout settles it, and the physics is not
touched either way. The common prefix is matched directly; past MAX_CELLS table cells the
rest is matched only by kind, in order.
"""
from __future__ import annotations

import numpy as np

TOLERANCE = 0.2
JUMP_COST = 1.5
MAX_CELLS = 40_000_000

_MATCH, _DELETE, _INSERT = 0, 1, 2


def volume_alignment(old_kind, old_energy, new_kind, new_energy) -> np.ndarray:
    """Per new volume the old volume it continues, or -1."""
    old_kinds, new_kinds = np.asarray(old_kind).ravel(), np.asarray(new_kind).ravel()
    old_energies = np.asarray(old_energy, dtype=float).ravel()
    new_energies = np.asarray(new_energy, dtype=float).ravel()
    origin = np.full(len(new_kinds), -1, dtype=np.int64)
    prefix = _prefix(old_kinds, old_energies, new_kinds, new_energies)
    origin[:prefix] = np.arange(prefix)
    rest_old, rest_new = len(old_kinds) - prefix, len(new_kinds) - prefix
    if rest_old == 0 or rest_new == 0:
        return origin
    if rest_old * rest_new > MAX_CELLS:
        paired = _by_kind(old_kinds[prefix:], new_kinds[prefix:])
        origin[prefix:] = np.where(paired >= 0, paired + prefix, -1)
        return origin
    matched = _aligned(old_kinds[prefix:], old_energies[prefix:], new_kinds[prefix:], new_energies[prefix:])
    origin[prefix:] = np.where(matched >= 0, matched + prefix, -1)
    return origin


def _prefix(old_kinds, old_energies, new_kinds, new_energies) -> int:
    """How many leading volumes keep their kind and nearly their energy."""
    common = min(len(old_kinds), len(new_kinds))
    larger = np.maximum(np.maximum(np.abs(old_energies[:common]), np.abs(new_energies[:common])), 1e-300)
    same = (old_kinds[:common] == new_kinds[:common]) & (
        np.abs(old_energies[:common] - new_energies[:common]) / larger < TOLERANCE)
    breaks = np.flatnonzero(~same)
    return int(breaks[0]) if len(breaks) else common


def _aligned(old_kinds, old_energies, new_kinds, new_energies) -> np.ndarray:
    """The edit-distance alignment; per new volume its old index or -1."""
    old_count, new_count = len(old_kinds), len(new_kinds)
    columns = np.arange(new_count + 1, dtype=float)
    previous = columns.copy()
    choice = np.zeros((old_count + 1, new_count + 1), dtype=np.int8)
    choice[0, 1:] = _INSERT
    for row in range(1, old_count + 1):
        match_cost = np.minimum(np.abs(new_energies - old_energies[row - 1])
                                / np.maximum(np.maximum(np.abs(new_energies), abs(old_energies[row - 1])), 1e-300)
                                / TOLERANCE, JUMP_COST)
        match_cost[new_kinds != old_kinds[row - 1]] = np.inf
        through = np.empty(new_count + 1)
        through[0] = previous[0] + 1.0
        through[1:] = np.minimum(previous[1:] + 1.0, previous[:-1] + match_cost)
        base_choice = np.full(new_count + 1, _DELETE, dtype=np.int8)
        base_choice[1:][previous[:-1] + match_cost < previous[1:] + 1.0] = _MATCH
        current = np.minimum.accumulate(through - columns) + columns
        row_choice = np.where(current < through - 1e-12, _INSERT, base_choice).astype(np.int8)
        choice[row] = row_choice
        previous = current
    matched = np.full(new_count, -1, dtype=np.int64)
    row, column = old_count, new_count
    while row > 0 and column > 0:
        step = choice[row, column]
        if step == _MATCH:
            matched[column - 1] = row - 1
            row, column = row - 1, column - 1
        elif step == _DELETE:
            row -= 1
        else:
            column -= 1
    return matched


def _by_kind(old_kinds, new_kinds) -> np.ndarray:
    """In-order pairing by kind alone: the k-th new volume of a kind continues the k-th old one."""
    matched = np.full(len(new_kinds), -1, dtype=np.int64)
    for kind in np.unique(new_kinds):
        olds, news = np.flatnonzero(old_kinds == kind), np.flatnonzero(new_kinds == kind)
        common = min(len(olds), len(news))
        matched[news[:common]] = olds[:common]
    return matched
