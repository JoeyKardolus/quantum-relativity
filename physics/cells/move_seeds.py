"""Move seeds: the working positions follow their cells' shapes and the pull across squeezed joints.

Contract: `move_seeds(positions, tessellation, joints, squeeze, side, limit)` returns
new positions (n, 3) wrapped into [0, side)^3, for the cells the tessellation was made of:
  relax   each position moves RELAX (0.5) of the way to its cell's centroid, so cells
          stay round and their sizes follow the weights (fit_weights.py)
  draw    across every joint (a, b) the two positions move toward each other by
          DRAW (0.05) * squeeze * their nearest-image distance, each half of it: pull
          draws neighbours together across a joint it squeezed
  cap     each position's whole move (relax plus draw) is cut to at most `limit`
          times its cell's width (tessellated volume ** (1/3)), keeping its direction,
          so a step moves a seed only a little and the cells keep their neighbours;
          an empty cell's seed stays. `limit` >= 0 (np.inf: no cap).
Positions are labels for computing shapes; no rule reads them.
"""
from __future__ import annotations

import numpy as np

from physics.cells.tessellation import Tessellation

RELAX = 0.5
DRAW = 0.05


def move_seeds(positions: np.ndarray, tessellation: Tessellation, joints: np.ndarray,
               squeeze: np.ndarray, side: float, limit: float) -> np.ndarray:
    """Positions relaxed toward their centroids and drawn together across squeezed joints, capped."""
    wrapped = np.mod(positions, side)
    relaxed = RELAX * (tessellation.centroid - wrapped)
    offset = wrapped[joints[:, 1]] - wrapped[joints[:, 0]]
    offset -= side * np.round(offset / side)
    pulled = (DRAW / 2) * squeeze[:, None] * offset
    count = len(positions)
    drawn = np.stack([np.bincount(joints[:, 0], pulled[:, axis], count)
                      - np.bincount(joints[:, 1], pulled[:, axis], count) for axis in range(3)],
                     axis=1)
    width = np.cbrt(tessellation.volume)
    longest = np.where(width > 0, limit * np.maximum(width, 1e-300), 0.0)
    return np.mod(wrapped + _capped(relaxed + drawn, longest), side)


def _capped(moves: np.ndarray, longest: np.ndarray) -> np.ndarray:
    """Each row of `moves` shortened to at most its `longest`, direction kept."""
    length = np.linalg.norm(moves, axis=1)
    scale = np.minimum(1.0, np.divide(longest, length, out=np.ones(len(length)),
                                      where=length > longest))
    return moves * scale[:, None]
