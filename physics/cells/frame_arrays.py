"""Frame arrays: what changes every frame, as small binary arrays for the page.

Contract: `frame_arrays(engine)` returns (corners, pull, push):
  corners  (k, 3) float32, the mesh's corner positions (mesh_message.py's order)
  pull     (n,) uint8 per cell, round(255 * pull / largest pull now) (Grid.pull):
           255 the cell pulling hardest now, 0 none; all 0 when nothing pulls
  push     (j,) uint8 per joint, round(255 * squeeze) (Grid.squeeze): 0 at or above
           rest thickness, 255 at the collapse limit
At 2000 cells that is about 13 000 corners: about 160 kB a frame.
"""
from __future__ import annotations

import numpy as np

from physics.cells.engine import Engine


def frame_arrays(engine: Engine) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Corner positions as float32, per-cell pull and per-joint push as uint8."""
    grid = engine.grid
    largest = float(grid.pull.max()) if grid.cell_count else 0.0
    pull = np.zeros(grid.cell_count) if largest <= 0 else grid.pull / largest
    return (np.ascontiguousarray(engine.corners, dtype=np.float32),
            np.round(255 * np.clip(pull, 0, 1)).astype(np.uint8),
            np.round(255 * np.clip(grid.squeeze, 0, 1)).astype(np.uint8))
