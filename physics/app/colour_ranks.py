"""The colour value per volume: its rank of pull (energy volume) or squeeze (joint).

Contract: `colour_ranks(state)` returns uint8 per volume, 0..255. Energy volumes are ranked
among energy volumes by pull, joints among joints by squeeze, each spread evenly over
0..255 (higher pulls or is squeezed harder), so a few extreme values cannot crowd everyone
else into one shade; ties keep index order. A kind with one volume gets 255.
Pull is the engine's `pull_potential(state)` (each energy volume's pull spread and fading
through the contacts); squeeze is how far a joint is pressed below the size its energy
wants, 1 - size / `joint_rest_size(energy)` (above 0 squeezed, below 0 swollen). Display
only: nothing here feeds the physics.
"""
from __future__ import annotations

import numpy as np

from physics.abab import joint_rest_size, pull_potential


def colour_ranks(state) -> np.ndarray:
    """Per volume its pull rank (energy volumes) or squeeze rank (joints), 0..255."""
    kinds = np.asarray(state.kind)
    energy = np.asarray(state.energy, dtype=float)
    size = np.asarray(state.size, dtype=float)
    pull = np.asarray(pull_potential(state), dtype=float)
    with np.errstate(all="ignore"):
        squeeze = 1.0 - size / np.asarray(joint_rest_size(energy), dtype=float)
    ranks = np.zeros(len(kinds), dtype=np.uint8)
    for kind, values in ((0, pull), (1, squeeze)):
        members = np.flatnonzero(kinds == kind)
        ranks[members] = _rank_bytes(np.nan_to_num(values[members], nan=0.0, posinf=np.finfo(float).max))
    return ranks


def _rank_bytes(values) -> np.ndarray:
    """Each value's rank among them, spread over 0..255."""
    if len(values) == 0:
        return np.zeros(0, dtype=np.uint8)
    places = np.empty(len(values), dtype=float)
    places[np.argsort(values, kind="stable")] = np.arange(len(values))
    if len(values) == 1:
        return np.full(1, 255, dtype=np.uint8)
    return np.round(255.0 * places / (len(values) - 1)).astype(np.uint8)
