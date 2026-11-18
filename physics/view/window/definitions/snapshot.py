"""Snapshot: what the window draws and titles, one moment of a source.

Contract: `Snapshot` is a frozen record:
  step                      steps since the last reset
  corners, colours, pattern the triangles to draw (`grid_triangles`)
  drawn                     cells drawn
  low, high                 the grid's bounding corners
  stats                     the source's latest stats
  steps_per_second          how fast the source last stepped
  paused, error             whether stepping is paused; the error that stopped it ("" for none)
  generation                rises on each reset, so the window can fit the view again
  layer                     the slice's lattice index along z; None for the whole grid
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Snapshot:
    """One moment of the source, ready to draw."""

    step: int
    corners: np.ndarray
    colours: np.ndarray
    pattern: np.ndarray
    drawn: int
    low: np.ndarray
    high: np.ndarray
    stats: dict
    steps_per_second: float
    paused: bool
    error: str
    generation: int
    layer: int | None
