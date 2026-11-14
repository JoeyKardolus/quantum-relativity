"""The live simulation: one network stepped under all rules on, plus its display layout.

Contract: `Simulation(joints, seed, unevenness, speed)` resets at once.
`reset(seed, joints)` builds the start (`start_network`: an even 3D ball made
slightly uneven by `unevenness`, no lump placed), sets `settings` to all rules
on with attraction scaled to the even ball and the balancing drift calibrated
on it (`scene_rules`, the one calibration), and lays the joints out for display
(`initial_layout` relaxed START_ITERATIONS times). Any rule setting changed
before is dropped; unevenness and speed are kept. Unevenness is a start
condition, so a new value shows on the next reset.
`apply(values)` sets the named controls (RuleSettings fields, "unevenness",
"speed"), each clamped to `ranges`; unknown names raise ValueError. A changed
rule applies from the next step.
`advance()` runs one `step` and carries display positions across joint
collapse (`carry_positions`, merged joint at the energy-weighted mean of its
parts); returns True when the connection list changed. Raises ValueError when
the rules fail or a length stops being finite and positive (the network is
then left as it was).
`layout()` relaxes the display positions FRAME_ITERATIONS times toward the
current lengths (warm start) and returns them centred, in units of the
start's mean connection length.
Steps and the layout draw from separate generators seeded from `seed`, so a
run is the same whatever the frame rate. Positions live only here, never on
the network.
"""
from __future__ import annotations

from dataclasses import fields, replace

import numpy as np

from physics.app.setting_ranges import setting_ranges
from physics.app.start_network import start_network
from physics.rules import RuleSettings, step
from physics.viewer.carry_positions import carry_positions
from physics.viewer.initial_layout import initial_layout
from physics.viewer.joint_energy import joint_energy
from physics.viewer.relax_layout import relax_layout
from physics.viewer.scene_rules import scene_rules

START_ITERATIONS = 60
FRAME_ITERATIONS = 6
RULE_NAMES = tuple(field.name for field in fields(RuleSettings))
INTEGER_NAMES = ("rewiring",)


class Simulation:
    """A network, its rule settings, seeded generators and display positions."""

    def __init__(self, joints: int = 2500, seed: int = 1, unevenness: float = 0.5,
                 speed: float = 10.0) -> None:
        self.unevenness = unevenness
        self.speed = speed
        self.reset(seed, joints)

    def reset(self, seed: int, joints: int) -> None:
        """Fresh uneven start, recalibrated settings and a new display layout."""
        even, self.network = start_network(joints, seed, self.unevenness)
        self.seed = seed
        self.settings = scene_rules(even, seed)
        self.ranges = setting_ranges(self.settings)
        self.step_rng = np.random.default_rng(seed)
        self.layout_rng = np.random.default_rng(seed + 1)
        self.steps = 0
        self.start_joints = self.network.joint_count
        self.start_mean_length = float(self.network.length.mean())
        self._positions = relax_layout(self.network, initial_layout(self.network),
                                       self.layout_rng, START_ITERATIONS)

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
                rules[name] = int(round(clamped)) if name in INTEGER_NAMES else clamped
            else:
                setattr(self, name, clamped)
        self.settings = replace(self.settings, **rules)

    def advance(self) -> bool:
        """One rules step; True when the connection list changed."""
        with np.errstate(all="ignore"):
            stepped = step(self.network, self.settings, self.step_rng)
        if not (np.all(np.isfinite(stepped.length)) and np.all(stepped.length > 0)):
            raise ValueError(f"step {self.steps + 1}: a connection length stopped being finite and positive")
        changed = not np.array_equal(stepped.pairs, self.network.pairs)
        self._positions = carry_positions(self._positions, joint_energy(self.network),
                                          stepped.merged_into, stepped.joint_count)
        self.network = stepped
        self.steps += 1
        return changed

    def layout(self) -> np.ndarray:
        """Display positions relaxed toward current lengths, centred, in start mean lengths."""
        self._positions = relax_layout(self.network, self._positions, self.layout_rng,
                                       FRAME_ITERATIONS)
        return (self._positions - self._positions.mean(axis=0)) / self.start_mean_length
