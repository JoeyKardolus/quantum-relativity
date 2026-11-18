"""Snapshot: what the window draws and titles, one moment of a source.

Contract: `Snapshot` is a frozen record:
  step                      steps since the last reset
  corners, colours, pattern the slice's triangles (`grid_triangles`)
  drawn                     cells drawn, the slice's and the blocks' together
  low, high                 the corners of the grid's outer frame
  stats                     the source's latest stats
  steps_per_second          how fast the source last stepped
  paused, error             whether stepping is paused; the error that stopped it ("" for none)
  generation                rises on each reset
  layer                     the slice's lattice index along z
  blocks                    None while only the slice shows; for the whole grid the
                            packed cells' (corners, colours, pattern) as blocks
  places                    lattice places a side
  limited                   a cell's size is further from the typical than is drawn
  left_out                  packed cells not drawn as blocks: too spread to lay out
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
    blocks: tuple | None = None
    places: int = 0
    limited: bool = False
    left_out: int = 0
