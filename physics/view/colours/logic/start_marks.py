"""Start marks: the two packings every later packing is read against.

Contract: `start_marks(kind, packing)` takes the kinds and packings of a fresh start
and returns `{ENERGY_VOLUME: mean packing of its energy volumes, JOINT: that of its
joints}` as floats. By the size law a joint holds less energy than a volume, so it is
longer and far less packed (millions of times at the defaults); one mark for both would
leave every joint clear for ever. Raises ValueError when a kind has no cell or its mean
is not above 0: then there is nothing to read a packing against.
"""
from __future__ import annotations

import numpy as np

from physics.view.layout.definitions.kinds import ENERGY_VOLUME, JOINT


def start_marks(kind: np.ndarray, packing: np.ndarray) -> dict:
    """The start's mean packing of each kind."""
    kind, packing = np.asarray(kind), np.asarray(packing, dtype=np.float64)
    marks = {}
    for one in (ENERGY_VOLUME, JOINT):
        mine = packing[kind == one]
        if not mine.size or not float(mine.mean()) > 0.0:
            raise ValueError(f"the start has no packing to read kind {one} against")
        marks[one] = float(mine.mean())
    return marks
