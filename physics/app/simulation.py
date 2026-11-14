"""The live simulation: one cell grid stepped under all rules on, plus its display layout.

Contract: `Simulation(cells, seed, unevenness, speed)` resets at once.
`reset(seed, cells)` builds the start (`build_grid`: energy slightly uneven
everywhere by `unevenness`, no lump placed), sets `settings` to all rules on
with the drift balanced on the even grid of the same seed (`balancing_drift`,
the one calibration), and lays the energy volumes out for display: build_grid's
display positions scaled so neighbours sit about one radius sum apart, then
`cell_layout` START_ITERATIONS times. Any rule setting changed before is
dropped; unevenness and speed are kept. Unevenness is a start condition, so a
new value shows on the next reset.
`apply(values)` sets the named controls (CellSettings fields, "unevenness",
"speed"), each clamped to `ranges`; unknown names raise ValueError. A changed
rule applies from the next step.
`advance()` runs one `step` and carries display positions across joint
collapse (`carry_positions` over merged_into, a merged volume at the
energy-weighted mean of its parts); returns True when the joint list changed.
Raises ValueError when an energy or volume stops being finite and positive
(the grid is then left as it was).
`layout()` relaxes the display positions FRAME_ITERATIONS times (warm start)
and returns them centred; `radius()` gives each volume's display radius.
Both are in units of the start's mean diameter. Positions live only here,
never on the Grid.
"""
from __future__ import annotations

from dataclasses import fields, replace

import numpy as np

from physics.app.cell_layout import cell_layout
from physics.app.setting_ranges import setting_ranges
from physics.cells import CellSettings, Grid, balancing_drift, build_grid, step
from physics.viewer.carry_positions import carry_positions

START_ITERATIONS = 40
FRAME_ITERATIONS = 4
RULE_NAMES = tuple(field.name for field in fields(CellSettings))


class Simulation:
    """A cell grid, its rule settings and the display positions of its energy volumes."""

    def __init__(self, cells: int = 2500, seed: int = 1, unevenness: float = 0.5,
                 speed: float = 10.0) -> None:
        self.unevenness = unevenness
        self.speed = speed
        self.reset(seed, cells)

    def reset(self, seed: int, cells: int) -> None:
        """Fresh uneven grid, recalibrated settings and a new display layout."""
        self.grid, start_positions = build_grid(cells, seed, self.unevenness)
        even, _ = build_grid(cells, seed, 0.0)
        self.settings = CellSettings.all_on(drift=balancing_drift(even, CellSettings.all_on()))
        self.ranges = setting_ranges(self.settings)
        self.seed = seed
        self.steps = 0
        self.start_cells = self.grid.cell_count
        self.start_total_volume = float(self.grid.volume.sum())
        radius = _radius(self.grid.volume)
        self.unit = 2.0 * float(radius.mean())
        self._positions = cell_layout(_scaled(start_positions, self.grid, radius), self.grid.joints,
                                      radius, START_ITERATIONS)

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

    def advance(self) -> bool:
        """One rules step; True when the joint list changed."""
        with np.errstate(all="ignore"):
            stepped = step(self.grid, self.settings)
        for name in ("energy", "volume"):
            values = getattr(stepped, name)
            if not (np.all(np.isfinite(values)) and np.all(values > 0)):
                raise ValueError(f"step {self.steps + 1}: an {name} stopped being finite and positive")
        changed = (stepped.cell_count != self.grid.cell_count
                   or not np.array_equal(stepped.joints, self.grid.joints))
        self._positions = carry_positions(self._positions, self.grid.energy,
                                          stepped.merged_into, stepped.cell_count)
        self.grid = stepped
        self.steps += 1
        return changed

    def layout(self) -> np.ndarray:
        """Display positions relaxed toward touching neighbours, centred, in start diameters."""
        self._positions = cell_layout(self._positions, self.grid.joints, _radius(self.grid.volume),
                                      FRAME_ITERATIONS)
        return (self._positions - self._positions.mean(axis=0)) / self.unit

    def radius(self) -> np.ndarray:
        """Display radius per energy volume, in start diameters."""
        return _radius(self.grid.volume) / self.unit


def _radius(volume: np.ndarray) -> np.ndarray:
    return np.cbrt(3.0 * volume / (4.0 * np.pi))


def _scaled(positions: np.ndarray, grid: Grid, radius: np.ndarray) -> np.ndarray:
    """build_grid's positions scaled so the median joint is as long as the median radius sum."""
    joints = grid.joints
    if not len(joints):
        return np.array(positions, dtype=float)
    length = np.linalg.norm(positions[joints[:, 0]] - positions[joints[:, 1]], axis=1)
    wanted = radius[joints[:, 0]] + radius[joints[:, 1]]
    return np.array(positions, dtype=float) * float(np.median(wanted) / max(np.median(length), 1e-300))
