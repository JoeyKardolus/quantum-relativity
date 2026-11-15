"""The "stats" message: the engine's stats as plain JSON, the step and whether it runs.

Contract: `stats_message(simulation, running)` returns {"type": "stats", "step",
"running", "stats"}. "stats" is `physics.abab.stats(state, settings)` with every value plain JSON:
a bool stays a bool, an integer an int, a float is rounded to 5 decimals and a value that
is not finite becomes null. Sent with every frame and on start and pause, so every open
page shows the same run state.
"""
from __future__ import annotations

import math

import numpy as np

from physics.abab import stats
from physics.app.simulation import Simulation


def stats_message(simulation: Simulation, running: bool) -> dict:
    """The current state's stats, its step and the run state."""
    plain = {str(name): _plain(value) for name, value in stats(simulation.state, simulation.settings).items()}
    return {"type": "stats", "step": simulation.steps, "running": bool(running), "stats": plain}


def _plain(value):
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    number = float(value)
    return round(number, 5) if math.isfinite(number) else None
