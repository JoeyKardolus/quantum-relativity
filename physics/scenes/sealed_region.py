"""Sealed region: the black hole round the most packed energy volume, measured in Planck boxes.

Contract: `sealed_region(universe, settings)` reads universe 0 of a physics.engine State
(physics.engine cells) and takes the core: the energy volume with the most whole quanta
per box. Its joint is the joint boxes on its faces (the engine keeps no joint labels, so
"its joints" are the minus boxes it touches across a face). It is sealed when it has
such boxes and every one is squeezed to settings.seal_squeeze or past it (squeeze =
|quanta| / joint_rest - 1: nothing crosses out, as stats counts sealed lumps). Returns
a dict of floats:
  sealed             1.0 when the core is sealed, else 0.0
  sealed_boxes       boxes of the core and its joint boxes when sealed, else 0
  sealed_radius      radius of a ball of that many boxes, (3 * boxes / (4 pi))**(1/3)
  sealed_energy      whole quanta in those boxes when sealed, else 0
  collapsed_energy   the core's whole quanta (what fell together into it)
  core_boxes         the core's box count
  core_labels        how many start energy volumes' boxes it holds (start_volumes: the
                     plus groups of the start, labelled the same way; 1 until joints
                     collapsed between volumes; 0 without a start given)
  core_compression_min   the least squeezed of its joint boxes' squeeze (0 with none)
`sealed_region(universe, settings, start_volumes=None)`: start_volumes is the start's
owner array (cells(...)["owner"] at the start) used for core_labels.
The boxes are fixed Planck boxes; a box count is a volume of space here.
"""
from __future__ import annotations

import math

import numpy as np
from scipy import ndimage

from physics.engine import Settings, State, cells, host_boxes


def sealed_region(universe: State, settings: Settings, start_volumes: np.ndarray | None = None) -> dict:
    """Seal, boxes, ball radius and quanta round the most packed energy volume, and the core's quanta."""
    boxes = host_boxes(universe)
    quanta = boxes["quanta"].astype(np.int64)
    found = cells(boxes)
    count = found["volumes"]
    if count == 0:
        return {"sealed": 0.0, "sealed_boxes": 0.0, "sealed_radius": 0.0, "sealed_energy": 0.0,
                "collapsed_energy": 0.0, "core_boxes": 0.0, "core_labels": 0.0, "core_compression_min": 0.0}
    packing = found["quanta"][:count] / np.maximum(found["boxes"][:count], 1)
    core = int(np.argmax(packing))
    core_boxes = found["owner"] == core
    around = ndimage.binary_dilation(core_boxes) & (quanta < 0)
    squeezed = np.abs(quanta[around]) / settings.joint_rest - 1.0
    sealed = bool(around.any()) and bool(np.all(squeezed >= settings.seal_squeeze))
    region = core_boxes | around
    region_boxes = float(region.sum()) if sealed else 0.0
    labels = 0.0
    if start_volumes is not None:
        held = start_volumes[core_boxes]
        labels = float(np.unique(held[(held >= 0)]).size)
    return {"sealed": 1.0 if sealed else 0.0, "sealed_boxes": region_boxes,
            "sealed_radius": (3.0 * region_boxes / (4.0 * math.pi)) ** (1.0 / 3.0),
            "sealed_energy": float(np.abs(quanta[region]).sum()) if sealed else 0.0,
            "collapsed_energy": float(found["quanta"][core]), "core_boxes": float(found["boxes"][core]),
            "core_labels": labels, "core_compression_min": float(squeezed.min()) if squeezed.size else 0.0}
