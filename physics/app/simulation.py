"""The live simulation: one ABAB state stepped under all rules on, plus its drawing positions.

Contract: `Simulation(volumes, seed, unevenness, speed)` resets at once.
`reset(seed, volumes)` builds the start (`physics.abab.new_state(volumes, seed,
unevenness)`: the irregular ABAB structure, energy slightly uneven everywhere, no lump
placed; it returns the state and the building positions) and sets `settings` to all rules
on at their defaults (`Settings.all_on()`). Any rule setting changed before is dropped;
unevenness and speed are kept (unevenness shows on the next reset). The building
positions, scaled to the drawing (`fitted_positions`) and settled (START_ITERATIONS
rounds of `settle_positions`), become `positions`.
`apply(values)` sets the named controls (`rule_names`, "unevenness", "speed"), each
clamped to `ranges` (an integer rule stays an integer); unknown names raise ValueError. A
changed rule applies from the next step.
`advance()` runs one pure `physics.abab.step(state, settings)`. Raises ValueError when an
energy or a size stops being finite and positive (that state is then not taken). When the
topology changed (mesh_key) the positions are carried over (`carry_positions`); then
FRAME_ITERATIONS rounds of the layout run, warm-started.
`positions` (float64 (n, 3), one per volume) exist only for drawing: they never enter
the physics, and the state never sees them.
"""
from __future__ import annotations

from dataclasses import fields, replace

import numpy as np

from physics.abab import Settings, new_state, step
from physics.app.carry_positions import carry_positions
from physics.app.fitted_positions import fitted_positions
from physics.app.mesh_key import mesh_key
from physics.app.setting_ranges import setting_ranges
from physics.app.settle_positions import settle_positions
from physics.app.volume_radius import volume_radius

START_ITERATIONS = 30
FRAME_ITERATIONS = 4


class Simulation:
    """An ABAB state, its rule settings, its drawing positions and how many steps it has run."""

    def __init__(self, volumes: int = 2000, seed: int = 1, unevenness: float = 0.5,
                 speed: float = 10.0) -> None:
        self.unevenness = unevenness
        self.speed = speed
        self.reset(seed, volumes)

    def reset(self, seed: int, volumes: int) -> None:
        """Fresh uneven structure, default settings, settled drawing positions."""
        self.state, building = new_state(volumes, seed, self.unevenness)
        radii = volume_radius(self.state.size)
        start = fitted_positions(building, self.state.contacts, radii)
        self.positions = settle_positions(start, self.state.contacts, radii, START_ITERATIONS)
        self.settings = Settings.all_on()
        self.ranges = setting_ranges(self.settings)
        self.rule_names = tuple(field.name for field in fields(self.settings) if field.name in self.ranges)
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
            if name in self.rule_names:
                rules[name] = round(clamped) if isinstance(getattr(self.settings, name), int) else clamped
            else:
                setattr(self, name, clamped)
        self.settings = replace(self.settings, **rules)

    def advance(self) -> None:
        """One engine step under the current settings, then the drawing follows."""
        with np.errstate(all="ignore"):
            stepped = step(self.state, self.settings)
        for name in ("energy", "size"):
            values = np.asarray(getattr(stepped, name), dtype=float)
            if not (np.all(np.isfinite(values)) and np.all(values > 0)):
                raise ValueError(f"step {self.steps + 1}: a {name} stopped being finite and positive")
        positions = self.positions
        if mesh_key(stepped) != mesh_key(self.state):
            positions = carry_positions(self.state, stepped, positions)
        self.positions = settle_positions(positions, stepped.contacts, volume_radius(stepped.size),
                                          FRAME_ITERATIONS)
        self.state = stepped
        self.steps += 1
