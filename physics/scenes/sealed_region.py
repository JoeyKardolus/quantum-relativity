"""Sealed region: the part of the grid round the most packed energy volume that nothing can leave.

Contract: `sealed_region(universe, settings)` reads the engine's relations (way_list)
and returns a dict of plain numbers:
  core            the id of the energy volume holding the most quanta; core_quanta
  core_joints     joints carrying a way from the core; core_sealed how many of them are
                  sealed: settings.seal_thin times the start's mean joint quanta or more
                  (cross.py lets nothing through such a joint)
  sealed          1 when the region is closed: starting at the core and stepping from
                  volume to volume through every joint that is not sealed does not reach
                  every volume
  volumes         energy volumes inside the region (the core and all reached that way);
                  0 when not sealed
  radius          the region's radius in lengths: half the cube root of the summed cubed
                  lengths of its volumes (0 when not sealed)
  energy          plus quanta inside the region (0 when not sealed)
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import breadth_first_order

from physics.engine import ENERGY_VOLUME, JOINT, Settings, State, way_list


def sealed_region(universe: State, settings: Settings) -> dict:
    """The most packed volume and the region round it that sealed joints close off."""
    held = universe.host("cell_quanta").astype(np.float64)
    kind = universe.kind_h[:universe.count]
    plus = np.where(kind == ENERGY_VOLUME, held, 0.0)
    core = int(np.argmax(plus))
    joints, first, second = way_list(universe)
    tight = held[joints] >= settings.seal_thin * universe.start_quanta[JOINT]
    mine = (first == core) | (second == core)
    passable = coo_matrix((np.ones(int((~tight).sum())), (first[~tight], second[~tight])),
                          shape=(universe.count, universe.count)).tocsr()
    reached = breadth_first_order(passable, core, directed=False, return_predecessors=False)
    closed = len(reached) < int((kind == ENERGY_VOLUME).sum())
    length = universe.host("length").astype(np.float64)[reached]
    return {"core": core, "core_quanta": int(plus[core]), "core_joints": int(len(np.unique(joints[mine]))),
            "core_sealed": int(len(np.unique(joints[mine & tight]))), "sealed": int(closed),
            "volumes": len(reached) if closed else 0,
            "radius": float(0.5 * np.cbrt((length ** 3).sum())) if closed else 0.0,
            "energy": int(plus[reached].sum()) if closed else 0}
