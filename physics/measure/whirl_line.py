"""Whirl line: where a whirl runs and which way, from the phase windings of the squares it pierces.

Contract: `whirl_line(windings, near=None, reach=None)` takes phase_winding.py's
(3, nx, ny, nz) windings and, optionally, a point (3 floats, box units, box i's
centre at i) and a distance: only squares whose centres lie within `reach` of `near`
count (both None: every square). A pierced square sits at its lowest corner plus 0.5
along its two in-plane axes. Returns a dict:
  found           1 when at least one counted square is pierced, else 0
  centre          the pierced squares' centre, weighted by |winding| (3 floats; NaN if none)
  winding_vector  sum over counted squares of winding times the square's normal: a
                  straight whirl of n turns along z through L layers gives (0, 0, n L)
  axis            winding_vector over its length, the whirl's direction (NaN if it is 0)
  pierced         number of counted pierced squares (the whirl's length in squares)
"""
from __future__ import annotations

import numpy as np


def whirl_line(windings: np.ndarray, near=None, reach: float | None = None) -> dict:
    """Centre, direction and length of the whirl the pierced squares near a point trace."""
    places, weights, vector = [], [], np.zeros(3)
    for axis in range(3):
        pierced = np.argwhere(windings[axis] != 0)
        if pierced.size == 0:
            continue
        centres = pierced.astype(np.float64)
        centres[:, (axis + 1) % 3] += 0.5
        centres[:, (axis + 2) % 3] += 0.5
        turns = windings[axis][tuple(pierced.T)]
        if near is not None and reach is not None:
            close = np.linalg.norm(centres - np.asarray(near, dtype=np.float64), axis=1) <= reach
            centres, turns = centres[close], turns[close]
        places.append(centres)
        weights.append(np.abs(turns))
        vector[axis] += turns.sum()
    places = np.concatenate(places) if places else np.zeros((0, 3))
    weights = np.concatenate(weights) if weights else np.zeros(0)
    found = int(weights.size > 0)
    centre = (places * weights[:, None]).sum(axis=0) / weights.sum() if found else np.full(3, np.nan)
    length = np.linalg.norm(vector)
    return {"found": found, "centre": centre, "winding_vector": vector,
            "axis": vector / length if length > 0 else np.full(3, np.nan), "pierced": int(weights.size)}
