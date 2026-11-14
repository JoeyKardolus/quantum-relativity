"""The "settings" message: every control's current value and allowed range.

Contract: `settings_message(simulation)` returns {"type": "settings",
"values": {name: {"value", "min", "max"}}} for every EngineSettings field plus
"unevenness" and "speed", in EngineSettings order then those two.
"""
from __future__ import annotations

from physics.app.simulation import RULE_NAMES, Simulation


def settings_message(simulation: Simulation) -> dict:
    """Current value and range of each control."""
    values = {}
    for name, (lowest, highest) in simulation.ranges.items():
        source = simulation.settings if name in RULE_NAMES else simulation
        values[name] = {"value": float(getattr(source, name)), "min": lowest, "max": highest}
    ordered = [*RULE_NAMES, "unevenness", "speed"]
    return {"type": "settings", "values": {name: values[name] for name in ordered}}
