"""The "frame" message: display positions, sizes, shades and stats for the current step.

Contract: `frame_message(simulation)` relaxes the layout once (`layout`) and
returns {"type": "frame", "step", "positions": [x, y, z, ...] per energy
volume (3 decimals, display only), "radius": per energy volume (display
scale, 3 decimals), "energy": per energy volume, energy per volume 0..1
(`energy_shades`), "squeeze": per joint 0..1 of the push limit
(`squeeze_shades`), "stats": every `grid_stats` key (collapsed counted from
the reset) plus "collapses", the same count under the frame's own name}.
Numbers are plain JSON: a value that is not finite becomes null, a float is
rounded to 5 decimals.
"""
from __future__ import annotations

import math

import numpy as np

from physics.app.energy_shades import energy_shades
from physics.app.simulation import Simulation
from physics.app.squeeze_shades import squeeze_shades
from physics.cells import grid_stats


def frame_message(simulation: Simulation) -> dict:
    """One frame of the message contract for the simulation's current grid."""
    grid = simulation.grid
    stats = {name: _plain(value) for name, value in
             grid_stats(grid, simulation.start_total_volume,
                        simulation.start_cells).items()}
    stats["collapses"] = simulation.start_cells - int(grid.cell_count)
    return {
        "type": "frame",
        "step": simulation.steps,
        "positions": np.round(simulation.layout(), 3).ravel().tolist(),
        "radius": np.round(simulation.radius(), 3).tolist(),
        "energy": energy_shades(grid),
        "squeeze": squeeze_shades(grid, simulation.settings),
        "stats": stats,
    }


def _plain(value):
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    number = float(value)
    return round(number, 5) if math.isfinite(number) else None
