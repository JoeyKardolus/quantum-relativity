"""Leak: every energy volume leaks energy along its open seams to the volumes around them, by pull (rule 6).

Contract: `leak(joints, seams, energy, pull, openness, rate)` returns the new
energies. seams (m, 3) are the energy volumes around each seam (Grid.seams; every
pair of a seam is a joint), openness (m,) each seam's width in [0, 1]
(seam_openness.py). pull (n,) is per energy volume the pull it exerts (move_space
passes attraction * p / 2, p = pull_potential; any positive scale gives the same
result, it cancels below). Energy moves only along seams, never through the
middle of a joint:
  outflow_a  = mean openness of the seams around a (0 with no seam)
  passage_ab = mean openness of the seams joint (a, b) meets (0 with no seam)
  given_a    = rate * E_a * outflow_a
  share_a->b = pull_b * passage_ab / sum_{k neighbour of a} pull_k * passage_ak
With every seam relaxed (openness 1) this is rate * E_a split by pull_b / sum pull_k,
not winner-takes-all; a neighbour reached only through closed seams gets nothing,
and a volume all of whose seams are closed gives nothing. Each unit that leaves
one volume lands in exactly one neighbour, so the total energy is kept exactly
(to rounding). A volume whose open neighbours all pull 0 keeps its energy. Drift
is not a rule of its own: it is the growth of the volumes this leak drains.
Raises ValueError for a rate outside [0, 1], a negative pull, or an openness
outside [0, 1] or not one per seam.
"""
from __future__ import annotations

import numpy as np

from physics.cells.seam_joints import seam_joints


def leak(joints: np.ndarray, seams: np.ndarray, energy: np.ndarray, pull: np.ndarray,
         openness: np.ndarray, rate: float) -> np.ndarray:
    """Energies after every energy volume leaks along its open seams to its neighbours by pull."""
    if not 0 <= rate <= 1:
        raise ValueError("leak must be in [0, 1]: a volume cannot give more than it holds")
    if (pull < 0).any():
        raise ValueError("pull must be >= 0 to share the leak")
    if len(openness) != len(seams) or ((openness < 0) | (openness > 1)).any():
        raise ValueError("openness must be one value in [0, 1] per seam")
    count = len(energy)
    each_corner = np.repeat(openness, 3)
    passage = _mean_over(seam_joints(joints, seams, count).ravel(), each_corner, len(joints))
    outflow = _mean_over(seams.ravel(), each_corner, count)
    heads, tails = joints[:, 0], joints[:, 1]
    toward_tails = pull[tails] * passage
    toward_heads = pull[heads] * passage
    around = np.bincount(heads, toward_tails, count) + np.bincount(tails, toward_heads, count)
    per_weight = np.divide(rate * energy * outflow, around, out=np.zeros(count), where=around > 0)
    to_tails = per_weight[heads] * toward_tails
    to_heads = per_weight[tails] * toward_heads
    given = np.bincount(heads, to_tails, count) + np.bincount(tails, to_heads, count)
    taken = np.bincount(tails, to_tails, count) + np.bincount(heads, to_heads, count)
    return energy - given + taken


def _mean_over(owner: np.ndarray, value: np.ndarray, count: int) -> np.ndarray:
    """Per owner the mean of its values, 0 for an owner with none."""
    total = np.bincount(owner, minlength=count).astype(float)
    return np.divide(np.bincount(owner, value, count), total, out=np.zeros(count),
                     where=total > 0)
