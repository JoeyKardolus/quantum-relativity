"""Sealed region: the black hole around the most squeezed box, measured by box count.

Contract: `sealed_region(grid, settings)` finds the 6-connected region of boxes whose B
compression (B-energy / settings.b_rest) is at or past settings.seal_compression that
holds the most compressed box. Returns a dict of floats:
  sealed_boxes       boxes in that region (0 when no box is sealed)
  sealed_radius      radius of a ball of that many boxes, (3 * boxes / (4 pi))**(1/3)
  sealed_energy      A- plus B-energy inside the region
  collapsed_energy   A- plus B-energy in the region's boxes past collapse_compression
Box counts are the engine's fixed boxes; no length follows energy yet (bible, rule 4).
"""
from __future__ import annotations

import math

import numpy as np
from scipy import ndimage

from physics.fluid import Grid, Settings
from physics.scenes.box_energies import box_energies


def sealed_region(grid: Grid, settings: Settings) -> dict:
    """Boxes, ball radius, energy and collapsed energy of the sealed region round the core."""
    a_energy, b_energy = box_energies(grid)
    compression = b_energy / settings.b_rest
    labels, _ = ndimage.label(compression >= settings.seal_compression)
    core_label = labels[np.unravel_index(int(np.argmax(compression)), compression.shape)]
    if core_label == 0:
        return {"sealed_boxes": 0.0, "sealed_radius": 0.0, "sealed_energy": 0.0, "collapsed_energy": 0.0}
    region = labels == core_label
    energy = a_energy + b_energy
    boxes = float(region.sum())
    collapsed = region & (compression > settings.collapse_compression)
    return {"sealed_boxes": boxes,
            "sealed_radius": (3.0 * boxes / (4.0 * math.pi)) ** (1.0 / 3.0),
            "sealed_energy": float(energy[region].sum()),
            "collapsed_energy": float(energy[collapsed].sum())}
