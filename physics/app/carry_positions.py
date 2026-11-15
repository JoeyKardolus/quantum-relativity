"""Display positions carried across a step that changed who touches whom.

Contract: `carry_positions(old_state, new_state, old_positions)` returns float64 (n, 3)
display positions for `new_state`'s volumes, from `old_positions` (one per old volume):
  - a volume that continues an old one (`volume_alignment`) keeps its position;
  - joint collapse: the old joints that are gone, grouped with the energy volumes they
    touched (groups joined through shared energy volumes), put the one energy volume of a
    group that continues at the energy-weighted mean of the whole group's old positions;
  - a new volume (a split's second half, its new joint) goes to the mean of its placed
    contacts, nudged by NUDGE of its own radius along a fixed per-index direction so two
    new volumes never land on one point, repeated until every volume is placed (any left
    without placed contacts goes to the mean of all placed ones).
The layout (`settle_positions`) then sorts out the rest over the next frames. Reads only
`kind`, `energy`, `size` and `contacts` of the states; never changes them.
"""
from __future__ import annotations

import numpy as np

from physics.app.volume_alignment import volume_alignment
from physics.app.volume_radius import volume_radius

NUDGE = 0.4
_GOLDEN = np.pi * (3.0 - np.sqrt(5.0))


def carry_positions(old_state, new_state, old_positions) -> np.ndarray:
    """New display positions: kept, collapsed to the energy-weighted mean, or placed among contacts."""
    old_points = np.asarray(old_positions, dtype=float).reshape(-1, 3)
    origin = volume_alignment(old_state.kind, old_state.energy, new_state.kind, new_state.energy)
    placed = np.full((len(origin), 3), np.nan)
    kept = origin >= 0
    placed[kept] = old_points[origin[kept]]
    _collapse_to_mean(old_state, old_points, origin, placed)
    _place_new(new_state, placed)
    return placed


def _collapse_to_mean(old_state, old_points, origin, placed) -> None:
    """Put each collapse's surviving energy volume at its group's energy-weighted mean."""
    kinds = np.asarray(old_state.kind)
    energy = np.asarray(old_state.energy, dtype=float)
    pairs = np.asarray(old_state.contacts, dtype=np.int64).reshape(-1, 2)
    continued = np.zeros(len(kinds), dtype=bool)
    continued[origin[origin >= 0]] = True
    gone_joints = np.flatnonzero((kinds == 1) & ~continued)
    if len(gone_joints) == 0:
        return
    group = np.arange(len(kinds))
    for joint in gone_joints:
        touching = pairs[(pairs == joint).any(axis=1)].ravel()
        members = np.unique(touching)
        roots = np.unique([_root(group, member) for member in members])
        for root in roots[1:]:
            group[root] = roots[0]
    roots = np.array([_root(group, volume) for volume in range(len(kinds))])
    new_of_old = np.full(len(kinds), -1, dtype=np.int64)
    new_of_old[origin[origin >= 0]] = np.flatnonzero(origin >= 0)
    for root in np.unique(roots[gone_joints]):
        members = np.flatnonzero(roots == root)
        volumes = members[kinds[members] == 0]
        survivors = new_of_old[volumes][new_of_old[volumes] >= 0]
        if len(survivors) != 1 or energy[volumes].sum() <= 0:
            continue
        weights = energy[volumes] / energy[volumes].sum()
        placed[survivors[0]] = weights @ old_points[volumes]


def _root(group, volume) -> int:
    """The group a volume belongs to (union-find root, with path halving)."""
    while group[volume] != volume:
        group[volume] = group[group[volume]]
        volume = group[volume]
    return int(volume)


def _place_new(new_state, placed) -> None:
    """Fill every unplaced volume from its placed contacts, nudged apart, until all are placed."""
    pairs = np.asarray(new_state.contacts, dtype=np.int64).reshape(-1, 2)
    radius = volume_radius(new_state.size)
    while True:
        missing = np.isnan(placed[:, 0])
        if not missing.any():
            return
        sums, counts = np.zeros_like(placed), np.zeros(len(placed))
        for this, other in ((pairs[:, 0], pairs[:, 1]), (pairs[:, 1], pairs[:, 0])):
            usable = missing[this] & ~missing[other]
            np.add.at(sums, this[usable], placed[other[usable]])
            np.add.at(counts, this[usable], 1.0)
        ready = missing & (counts > 0)
        if not ready.any():
            fallback = placed[~missing].mean(axis=0) if (~missing).any() else np.zeros(3)
            ready = missing
            sums[ready], counts[ready] = fallback, 1.0
        indices = np.flatnonzero(ready)
        placed[indices] = sums[indices] / counts[indices, None] + _nudge(indices) * (NUDGE * radius[indices])[:, None]


def _nudge(indices) -> np.ndarray:
    """A fixed unit direction per volume index (a golden-angle spiral on the sphere)."""
    height = 1.0 - 2.0 * ((indices * 0.618033988749895) % 1.0)
    around = np.sqrt(np.maximum(0.0, 1.0 - height * height))
    angle = indices * _GOLDEN
    return np.stack([around * np.cos(angle), around * np.sin(angle), height], axis=1)
