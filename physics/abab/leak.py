"""Leak: energy moves from energy volume to energy volume only through a joint, A -> B -> A, by pull (rule 6).

Contract: `leak(state, pull, openness, rate)` returns the new energy per volume.
pull (n,) >= 0 per volume (only the energy volumes' values are read), openness (n,)
in [0, 1] per volume (only the joints' values are read, joint_openness.py). Over
every passage (giver a, joint b, taker c; passages.py):
  outflow_a  = mean openness of the joints a touches (0 with none)
  given_a    = rate * E_a * outflow_a
  share_abc  = pull_c * open_b / sum over a's passages of pull_c' * open_b'
With every joint relaxed this is rate * E_a split by the takers' pull, not
winner-takes-all; a joint past the seal point (openness 0) passes nothing, and an
energy volume all of whose joints are sealed gives nothing. A joint is the passage,
not a store: its own energy is not changed. Each unit that leaves one energy volume
lands in exactly one other, so the total is kept to rounding. An energy volume whose
open passages all lead to pull 0 keeps its energy.
Raises ValueError for a rate outside [0, 1], a negative pull or an openness outside [0, 1].
"""
from __future__ import annotations

import numpy as np

from physics.abab.passages import passages
from physics.abab.state import State


def leak(state: State, pull: np.ndarray, openness: np.ndarray, rate: float) -> np.ndarray:
    """Energy per volume after every energy volume leaks through its open joints to the others, by pull."""
    if not 0 <= rate <= 1:
        raise ValueError("leak must be in [0, 1]: a volume cannot give more than it holds")
    if (pull < 0).any():
        raise ValueError("pull must be >= 0 to share the leak")
    if ((openness < 0) | (openness > 1)).any():
        raise ValueError("openness must be in [0, 1]")
    count = len(state.kind)
    volume, joint = state.contacts[:, 0], state.contacts[:, 1]
    touched = np.bincount(volume, minlength=count).astype(float)
    outflow = np.divide(np.bincount(volume, openness[joint], count), touched,
                        out=np.zeros(count), where=touched > 0)
    giver, through, taker = passages(state.contacts)
    weight = pull[taker] * openness[through]
    around = np.bincount(giver, weight, count)
    per_weight = np.divide(rate * state.energy * outflow, around, out=np.zeros(count), where=around > 0)
    moved = per_weight[giver] * weight
    return state.energy - np.bincount(giver, moved, count) + np.bincount(taker, moved, count)
