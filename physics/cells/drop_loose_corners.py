"""Drop loose corners: remove corners no face uses and renumber the rest.

Contract: `drop_loose_corners(corners, faces)` returns (corners, faces): only the
corners some face ring uses, in their old order, and the rings renumbered to them.
Positions are not changed.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np


def drop_loose_corners(corners: np.ndarray, faces: Sequence[np.ndarray]):
    """Corners some face uses, and the faces renumbered onto them."""
    used = np.zeros(len(corners), dtype=bool)
    used[np.concatenate(faces)] = True
    if used.all():
        return corners, list(faces)
    renumber = np.cumsum(used) - 1
    return corners[used], [renumber[ring] for ring in faces]
