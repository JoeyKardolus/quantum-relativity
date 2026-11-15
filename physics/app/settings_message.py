"""The "settings" message: every control's current value and allowed range.

Contract: `settings_message(simulation)` returns {"type": "settings",
"values": {name: {"value", "min", "max"}}} for every rule the page can set
(`simulation.rule_names`, in Settings order) then "unevenness" and "speed".
"""
from __future__ import annotations

from physics.app.simulation import Simulation


def settings_message(simulation: Simulation) -> dict:
    """Current value and range of each control."""
    values = {}
    for name in [*simulation.rule_names, "unevenness", "speed"]:
        lowest, highest = simulation.ranges[name]
        source = simulation.settings if name in simulation.rule_names else simulation
        values[name] = {"value": float(getattr(source, name)), "min": lowest, "max": highest}
    return {"type": "settings", "values": values}
