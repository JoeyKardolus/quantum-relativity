"""Leak: every energy volume leaks energy to its neighbours, harder pullers getting more (rule 6).

Contract: `leak(joints, energy, pull, rate)` returns the new energies. pull (n,)
is per energy volume the pull it exerts across each of its joints (move_space
passes attraction * p / 2, p = pull_potential; any positive scale gives the same
result, it cancels below). Volume a leaks rate * E_a, shared over its neighbours:
  share_a->b = pull_b / sum_{k neighbour of a} pull_k
so every neighbour with pull > 0 gets some and a neighbour pulling twice as hard
gets twice as much: not winner-takes-all. Equal pulls share evenly. Each unit
that leaves one volume lands in exactly one neighbour, so the total energy is
kept exactly (to rounding). A volume with no joint, or whose neighbours all pull
0, keeps its energy. Drift is not a rule of its own: it is the growth of the
volumes this leak drains (rest_volume follows the energy; move_space.py).
Raises ValueError for a rate outside [0, 1] or a negative pull.
"""
from __future__ import annotations

import numpy as np


def leak(joints: np.ndarray, energy: np.ndarray, pull: np.ndarray, rate: float) -> np.ndarray:
    """Energies after every energy volume leaks `rate` of its energy to its neighbours by pull."""
    if not 0 <= rate <= 1:
        raise ValueError("leak must be in [0, 1]: a volume cannot give more than it holds")
    if (pull < 0).any():
        raise ValueError("pull must be >= 0 to share the leak")
    heads, tails = joints[:, 0], joints[:, 1]
    count = len(energy)
    around = np.bincount(heads, pull[tails], count) + np.bincount(tails, pull[heads], count)
    per_pull = np.divide(rate * energy, around, out=np.zeros(count), where=around > 0)
    to_tails = per_pull[heads] * pull[tails]
    to_heads = per_pull[tails] * pull[heads]
    given = np.bincount(heads, to_tails, count) + np.bincount(tails, to_heads, count)
    taken = np.bincount(tails, to_tails, count) + np.bincount(heads, to_heads, count)
    return energy - given + taken
