"""Core run: one placed universe stepped to the end, with the readings a black hole measure needs.

Contract: `core_run(universe, middle, steps, window, every, settings)` takes a placed
start whose core (or, in the twin, plain middle volume) sits at lattice place (middle,
middle, middle), steps it `steps` times with every rule on and returns a dict:
  rows      int ids, shape (shells + 1, 6): the core and the volume 1 .. shells volumes
            out along each of the six rows through it; shells = side // 4
  pull      per shell, the pull there, the mean over the six rows and over the readings
            taken every `every` steps in the last `window` steps
  scatter   the standard deviation, over rows, shells 0 .. shells - 1 and readings, of
            the pull there less the pull at the same row's last shell: how uneven the
            pull is along the rows by itself
  entered   per boundary 0 .. shells, the plus quanta inside it (the core and the
            shells up to it, shell_ways) at the end less at the start of the window
  gathered  the same, at the end less at step 0
The universe is stepped in place. Reads `pull` and the volumes' quanta; the engine's
errors propagate.
"""
from __future__ import annotations

import numpy as np

from physics.engine import Grid, Settings, step
from physics.scenes.along_grid.definitions.row_cells import row_cells
from physics.scenes.black_hole.definitions.shell_ways import shell_ways
from physics.scenes.placed_start.definitions.place_cell import place_cell
from physics.scenes.volumes.logic.volume_field import volume_field


def _inside(universe: Grid, ways: np.ndarray, shells: int) -> np.ndarray:
    """Per boundary, the plus quanta of the start volumes inside it now."""
    held = volume_field(universe)[0]
    return np.array([held[ways <= shell].sum() for shell in range(shells + 1)])


def core_run(universe: Grid, middle: int, steps: int, window: int, every: int, settings: Settings) -> dict:
    """Step a universe to the end; the rows' pull over the window and the energy that entered every boundary."""
    side = universe.side
    shells = side // 4
    core = place_cell(side, (middle,) * 3)
    rows = np.stack([row_cells(side, core, axis, sign, 2 * shells)[::2] for axis in range(3) for sign in (1, -1)],
                    axis=1)
    ways = shell_ways(side // 2, middle // 2)
    pulls, inside = [], {}
    for taken in range(steps + 1):
        if taken:
            step(universe, settings)
        if taken in (0, steps - window, steps):
            inside[taken] = _inside(universe, ways, shells)
        if taken > steps - window and (steps - taken) % every == 0:
            pulls.append(universe.host("pull").astype(np.float64)[rows])
    pulls = np.array(pulls)
    below_far = pulls[:, :-1, :] - pulls[:, -1:, :]
    return {"rows": rows, "pull": pulls.mean(axis=(0, 2)), "scatter": float(below_far.std()),
            "entered": inside[steps] - inside[steps - window], "gathered": inside[steps] - inside[0]}
