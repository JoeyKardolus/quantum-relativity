"""Core numbers: what one black hole run measured, with the twin run subtracted.

Contract: `core_numbers(universe, read, twin, level, settings)` takes a universe at the
end of its run, its core_run readings, the twin's (the same start with no core) and the
depth level, and returns a dict of numbers and lists:
  core_energy   the core's quanta in even start volumes; core_clock its local time;
                core_gain the energy that went into the core over the window, in even
                start volumes (boundary 0, reported apart)
  distance      per shell, summed lengths from the core's middle to that shell's
                volume's middle, the mean of the six rows (middle_distances)
  inflow        per boundary, the energy that crossed it inward over the window, in
                even start volumes, twin subtracted; gathered the same since the start
  depth         per shell, how much lower the pull lies than in the twin, less the same
                at the last shell (so the last shell reads 0)
  held_volumes, held_radius, held_limit (1: the region reached the last shell it can)
  depth_radius, depth_limit (1: the depth is above the level out to the last shell)
  collapses, merges, tears   events in the run, counted and not followed
  far, box      the last shell's distance, and the summed lengths once round a row
                through the core (the mean of the three rows): for the law beside
Writes nothing.
"""
from __future__ import annotations

import numpy as np

from physics.engine import Grid, Settings
from physics.scenes.along_grid.definitions.row_cells import row_cells
from physics.scenes.along_grid.logic.middle_distances import middle_distances
from physics.scenes.black_hole.definitions.depth_radius import depth_radius
from physics.scenes.black_hole.definitions.held_shells import held_shells
from physics.scenes.run.definitions.even_volume_quanta import even_volume_quanta


def _shell_distances(universe: Grid, core: int, shells: int) -> np.ndarray:
    """Per shell, the summed lengths from the core's middle to the shell's volume's middle, the mean of the six rows."""
    rows = [middle_distances(universe, row_cells(universe.side, core, axis, sign, 2 * shells))[::2]
            for axis in range(3) for sign in (1, -1)]
    return np.mean(rows, axis=0)


def _once_round(universe: Grid, core: int) -> float:
    """The summed lengths once round a row through the core, the mean of the three rows."""
    length = universe.host("length").astype(np.float64)
    return float(np.mean([length[row_cells(universe.side, core, axis, 1, universe.side - 1)].sum()
                          for axis in range(3)]))


def core_numbers(universe: Grid, read: dict, twin: dict, level: float, settings: Settings) -> dict:
    """One run's numbers: the core, the flow through every boundary, the depth at every shell and the two radii."""
    even = even_volume_quanta(settings)
    shells = len(read["rows"]) - 1
    core = int(read["rows"][0, 0])
    distance = _shell_distances(universe, core, shells)
    inflow = (read["entered"] - twin["entered"]) / even
    lower = twin["pull"] - read["pull"]
    depth = lower - lower[-1]
    held = held_shells(inflow)
    radius, deep = depth_radius(depth, distance, level)
    return {"core_energy": float(universe.host("cell_quanta")[core]) / even,
            "core_clock": float(universe.host("clock")[core]), "core_gain": float(inflow[0]),
            "distance": distance.tolist(), "inflow": inflow.tolist(),
            "gathered": ((read["gathered"] - twin["gathered"]) / even).tolist(), "depth": depth.tolist(),
            "held_volumes": held, "held_radius": float(distance[held]) if held else 0.0,
            "held_limit": int(held == shells - 1), "depth_radius": radius, "depth_limit": int(deep == shells),
            "collapses": int(universe.collapses), "merges": int(universe.merges), "tears": int(universe.tears),
            "far": float(distance[-1]), "box": _once_round(universe, core)}
