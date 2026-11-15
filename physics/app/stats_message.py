"""The "stats" message: the engine's stats as plain JSON, the step and whether it runs.

Contract: `stats_message(simulation, running)` returns {"type": "stats", "step",
"running", "stats"}. "stats" is `physics.cells.engine_stats(engine)` with every value
plain JSON: a bool stays a bool, an integer an int, a float is rounded to 5 decimals
and a value that is not finite becomes null. Sent with every frame and on start and
pause, so every open page shows the same run state.
"""
from __future__ import annotations

import math

import numpy as np

from physics.app.simulation import Simulation
from physics.cells import engine_stats


def stats_message(simulation: Simulation, running: bool) -> dict:
    """The current engine's stats, its step and the run state."""
    stats = {name: _plain(value) for name, value in engine_stats(simulation.engine).items()}
    return {"type": "stats", "step": simulation.steps, "running": bool(running), "stats": stats}


def _plain(value):
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    number = float(value)
    return round(number, 5) if math.isfinite(number) else None
