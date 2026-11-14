"""The "shapes" message: the engine's cell and joint shapes as they are, plus its stats.

Contract: `shapes_message(simulation)` returns {"type": "shapes", "step",
"cells", "joints", "stats"}. "cells" and "joints" are passed on unchanged from
`physics.cells.shapes(engine)` (each cell {"id", "faces", "pull"}, each joint
{"a", "b", "polygon", "push", "thickness", "sealed"}; polygons are the
tessellation's own faces; "sealed" marks an outer wall of a sealed lump). "stats" is `physics.cells.engine_stats(engine)` with every value plain
JSON: a value that is not finite becomes null, a float is rounded to 5 decimals.
Nothing here computes or alters a shape.
"""
from __future__ import annotations

import math

import numpy as np

from physics.app.simulation import Simulation
from physics.cells import engine_stats, shapes


def shapes_message(simulation: Simulation) -> dict:
    """One shapes message for the simulation's current engine."""
    drawn = shapes(simulation.engine)
    stats = {name: _plain(value) for name, value in engine_stats(simulation.engine).items()}
    return {"type": "shapes", "step": simulation.steps, "cells": drawn["cells"],
            "joints": drawn["joints"], "stats": stats}


def _plain(value):
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    number = float(value)
    return round(number, 5) if math.isfinite(number) else None
