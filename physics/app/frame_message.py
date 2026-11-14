"""The "frame" message: display positions, energy shades and stats for the current step.

Contract: `frame_message(simulation)` relaxes the layout once (`layout`) and
returns {"type": "frame", "step", "positions": [x, y, z, ...] (3 decimals,
display only), "energy": per connection 0..1 (`energy_shades`), "stats":
{"joints", "collapsed" (joints lost to joint collapse since the reset),
"mean_length_vs_start" (null once no connection is left),
"energy_spread" (std / mean of connection energy),
"lumps" (`lump_count`)}}.
"""
from __future__ import annotations

import numpy as np

from physics.app.energy_shades import energy_shades
from physics.app.lump_count import lump_count
from physics.app.simulation import Simulation


def frame_message(simulation: Simulation) -> dict:
    """One frame of the message contract for the simulation's current network."""
    network = simulation.network
    energy = network.energy
    spread = float(energy.std() / energy.mean()) if len(energy) else 0.0
    size = round(float(network.length.mean()) / simulation.start_mean_length, 5) if len(energy) else None
    return {
        "type": "frame",
        "step": simulation.steps,
        "positions": np.round(simulation.layout(), 3).ravel().tolist(),
        "energy": energy_shades(network),
        "stats": {
            "joints": network.joint_count,
            "collapsed": simulation.start_joints - network.joint_count,
            "mean_length_vs_start": size,
            "energy_spread": round(spread, 5),
            "lumps": lump_count(network),
        },
    }
