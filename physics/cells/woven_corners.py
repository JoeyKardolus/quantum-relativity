"""Woven corners: the corners kept where they were, placed afresh where something changed or no longer fits.

Contract: `woven_corners(woven, built, kept, touched)` returns (k, 3) positions for the
corners of `woven` (cell_complex.py):
  built     (k, 3) where place_corners.py puts every corner for the skeleton and the
            joints' thicknesses now: always a fit, gap-free and the right way out
  kept      (k, 3) where each corner would rather be (where it was, or where it rides
            to with the skeleton's move, step.py); NaN where it has no such place (a
            corner new after a collapse or split)
  touched   (s,) bool per skeleton corner: a collapse or split changed or moved
            something there, so every corner near it (key column 1) is built afresh,
            never half old and half new (the thin cells between them would not fit)
  checked   a cell left smaller than MIN_SHARE (0.2) of its built size is a misfit:
            every skeleton corner of its place (cell_skeleton) becomes touched and the
            corners are kept again, up to EXPANSIONS (8) rounds; then every corner is
            built
Positions decide nothing in the grid; they only follow it.
"""
from __future__ import annotations

import numpy as np

from physics.cells.cell_complex import CellComplex
from physics.cells.cell_sizes import cell_sizes

MIN_SHARE = 0.2
EXPANSIONS = 8


def woven_corners(woven: CellComplex, built: np.ndarray, kept: np.ndarray, touched: np.ndarray) -> np.ndarray:
    """Corners kept where they would rather be, built afresh where touched, missing or misfitting."""
    count = len(woven.cell_kind)
    floor = MIN_SHARE * cell_sizes(built, woven.faces, woven.face_cells, count)
    touched = np.array(touched, dtype=bool)
    usable = np.isfinite(kept).all(axis=1)
    for _ in range(EXPANSIONS):
        keep = usable & ~touched[woven.keys[:, 1]]
        positions = np.where(keep[:, None], kept, built)
        misfit = cell_sizes(positions, woven.faces, woven.face_cells, count) < floor
        if not misfit.any():
            return positions
        places = woven.cell_skeleton[misfit]
        touched[places[places >= 0]] = True
    return built
