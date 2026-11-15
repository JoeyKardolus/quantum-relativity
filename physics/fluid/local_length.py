"""Local length: how long a box is, measured inside the structure, from the energy it holds.

Contract: `local_length(a, b, a_rest, b_rest)` takes numpy arrays a, b of A- and
B-energy per box and returns, per box, length = (a_rest + b_rest) / (a + b): 1 at the
start's mean energy per box, shorter where energy packs, longer where it thins, inf
in a box holding nothing. Bible rule 4, length = constant / energy (E = hc / lambda):
a box is a fixed set of energy volumes (volumes never travel, foundation 4), so the
energy of each volume in it is proportional to the box's energy and so is the
inverse of each volume's length. The dynamics count boxes, which is counting volumes
(rule 7: the pull fades with the number of volumes in between); lengths, distances,
volumes and dimension are measured from these (rule 1), never fed back as
coordinates. A region's size is the sum of its boxes' length cubed (stats.py).
"""
from __future__ import annotations

import numpy as np


def local_length(a: np.ndarray, b: np.ndarray, a_rest: float, b_rest: float) -> np.ndarray:
    """Per box, the start's mean energy per box over the energy it holds."""
    energy = np.asarray(a, dtype=np.float64) + np.asarray(b, dtype=np.float64)
    with np.errstate(divide="ignore"):
        return np.where(energy > 0, (a_rest + b_rest) / np.where(energy > 0, energy, 1.0), np.inf)
