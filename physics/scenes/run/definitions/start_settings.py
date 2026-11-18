"""Start settings: a fit candidate's start amounts put into the engine's settings.

Contract: `START_AMOUNTS` names the settings that say how a universe starts (the
engine's EnergySettings but for the step's length). `start_settings(settings, start)`
returns `settings` (default `Settings()`) with every amount named in the dict `start`
replaced; with no start it returns the settings unchanged. Raises ValueError for a name
that is not a start amount, and what `Settings` raises for a value it refuses.
"""
from __future__ import annotations

from dataclasses import replace

from physics.engine import Settings

START_AMOUNTS = ("volume_quanta", "joint_quanta", "unevenness", "slope", "inner_unevenness", "joint_unevenness")


def start_settings(settings: Settings | None = None, start: dict | None = None) -> Settings:
    """The settings with a candidate's start amounts in place of their own."""
    settings = Settings() if settings is None else settings
    if not start:
        return settings
    unknown = sorted(set(start) - set(START_AMOUNTS))
    if unknown:
        raise ValueError(f"not start amounts: {unknown}; a start may name {START_AMOUNTS}")
    return replace(settings, **start)
