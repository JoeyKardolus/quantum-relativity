"""Sealed region: the black hole round the densest energy volume, measured in Planck boxes.

Contract: `sealed_region(universe, settings)` refreshes the cell sums
(physics.potts.cell_sums) and takes the core: the living energy volume with the most
energy per box. Its joints are the joints it touches (physics.potts.contact_pairs). It
is sealed when it has joints and every one is squeezed to settings.seal_compression or
past it (nothing crosses out, as stats counts sealed lumps). Returns a dict of floats:
  sealed             1.0 when the core is sealed, else 0.0
  sealed_boxes       boxes of the core and its joints when sealed, else 0
  sealed_radius      radius of a ball of that many boxes, (3 * boxes / (4 pi))**(1/3)
  sealed_energy      energy in those boxes when sealed, else 0
  collapsed_energy   the core's energy (what fell together into it)
  core_boxes, core_labels   the core's box count and how many start energy volumes it
                     stands for (more than 1 after joints collapsed round it)
  core_compression_min   the least squeezed of its joints' compression (0 with none)
The boxes are fixed Planck boxes; a box count is a volume of space here.
"""
from __future__ import annotations

import math

import numpy as np

from physics.potts import ENERGY_VOLUME, JOINT, QUANTUM, Settings, Universe, cell_sums, contact_pairs


def sealed_region(universe: Universe, settings: Settings) -> dict:
    """Seal, boxes, ball radius and energy round the densest energy volume, and the core's energy."""
    cell_sums(universe, settings)
    owner = universe.owner.numpy()
    kind = universe.kind.numpy()
    boxes = universe.cell_boxes.numpy()
    held = universe.cell_energy.numpy() * QUANTUM
    volumes = (universe.alive.numpy() == 1) & (boxes > 0) & (kind == ENERGY_VOLUME)
    core = int(np.argmax(np.where(volumes, held / np.maximum(boxes, 1), -1.0)))
    low, high, _ = contact_pairs(owner)
    around = np.concatenate([high[low == core], low[high == core]])
    joints = np.unique(around[kind[around] == JOINT])
    squeezed = universe.cell_compression.numpy()[joints]
    sealed = bool(joints.size) and bool(np.all(squeezed >= settings.seal_compression))
    region = np.isin(owner, np.append(joints, core))
    region_boxes = float(region.sum()) if sealed else 0.0
    return {"sealed": 1.0 if sealed else 0.0, "sealed_boxes": region_boxes,
            "sealed_radius": (3.0 * region_boxes / (4.0 * math.pi)) ** (1.0 / 3.0),
            "sealed_energy": float(held[np.append(joints, core)].sum()) if sealed else 0.0,
            "collapsed_energy": float(held[core]), "core_boxes": float(boxes[core]),
            "core_labels": float(universe.labels.numpy()[core]),
            "core_compression_min": float(squeezed.min()) if joints.size else 0.0}
