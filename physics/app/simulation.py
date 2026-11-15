"""The live simulation: one cell engine stepped under all rules on.

Contract: `Simulation(cells, seed, unevenness, speed)` resets at once.
`reset(seed, cells)` builds the start (`new_engine`: energy slightly uneven
everywhere by `unevenness`, no lump placed) and sets `settings` to all rules on
at their defaults (`EngineSettings.all_on()`, leak included). Any rule setting
changed before is dropped; unevenness and speed are kept. Unevenness is a start
condition, so a new value shows on the next reset.
`apply(values)` sets the named controls (EngineSettings fields, "unevenness",
"speed"), each clamped to `ranges`; unknown names raise ValueError. A changed
rule applies from the next step.
`advance()` runs one engine step. The engine contract allows either a method
`engine.step(settings)` (returning the next Engine, or None when it stepped in
place) or a pure `physics.cells.step(engine, settings)`; the method wins when
the Engine has one. Raises ValueError when an energy stops being finite and
positive (a returned engine is then not taken; one that stepped in place
already changed). `start_box` is [[x, y, z], [x, y, z]], the bounding box of
`engine.corners` at the reset (the starting cube), kept through the run.
The app holds no geometry: corners, faces, pulls and pushes come from the engine
(`physics.cells.mesh_message`, `physics.cells.frame_arrays`).
"""
from __future__ import annotations

from dataclasses import fields, replace

import numpy as np

from physics import cells as cell_engine
from physics.app.setting_ranges import setting_ranges
from physics.cells import EngineSettings, new_engine

RULE_NAMES = tuple(field.name for field in fields(EngineSettings))


class Simulation:
    """A cell engine, its rule settings and how many steps it has run."""

    def __init__(self, cells: int = 1000, seed: int = 1, unevenness: float = 0.5,
                 speed: float = 10.0) -> None:
        self.unevenness = unevenness
        self.speed = speed
        self.reset(seed, cells)

    def reset(self, seed: int, cells: int) -> None:
        """Fresh uneven engine and default settings."""
        self.engine = new_engine(cells, seed, self.unevenness)
        corners = np.asarray(self.engine.corners, dtype=float).reshape(-1, 3)
        self.start_box = [corners.min(axis=0).tolist(), corners.max(axis=0).tolist()]
        self.settings = EngineSettings.all_on()
        self.ranges = setting_ranges(self.settings)
        self.seed = seed
        self.steps = 0

    def apply(self, values: dict[str, float]) -> None:
        """Set named controls, each clamped to its range."""
        unknown = set(values) - set(self.ranges)
        if unknown:
            raise ValueError(f"unknown settings: {sorted(unknown)}")
        rules = {}
        for name, value in values.items():
            lowest, highest = self.ranges[name]
            clamped = min(max(float(value), lowest), highest)
            if name in RULE_NAMES:
                rules[name] = clamped
            else:
                setattr(self, name, clamped)
        self.settings = replace(self.settings, **rules)

    def advance(self) -> None:
        """One engine step under the current settings."""
        with np.errstate(all="ignore"):
            stepped = _stepped(self.engine, self.settings)
        energy = getattr(stepped, "energy", None)
        if energy is not None:
            values = np.asarray(energy, dtype=float)
            if not (np.all(np.isfinite(values)) and np.all(values > 0)):
                raise ValueError(f"step {self.steps + 1}: an energy stopped being finite and positive")
        self.engine = stepped
        self.steps += 1


def _stepped(engine, settings: EngineSettings):
    """The next engine, through `engine.step(settings)` when it has one, else `physics.cells.step`."""
    method = getattr(engine, "step", None)
    if callable(method):
        result = method(settings)
        return engine if result is None else result
    return cell_engine.step(engine, settings)
