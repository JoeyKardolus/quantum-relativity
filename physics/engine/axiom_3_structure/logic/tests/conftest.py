"""Shared by the structure logic's tests: a small real start grid and the relation rules of an event by hand."""
from __future__ import annotations

import numpy as np
import pytest
import warp as wp

from physics.engine.axiom_3_structure.definitions.direction_codes import code_of_host, opposite_host, step_of_host
from physics.engine.axiom_3_structure.definitions.grid import Grid
from physics.engine.axiom_3_structure.definitions.kinds import ENERGY_VOLUME, JOINT, NO_CELL
from physics.engine.axiom_3_structure.definitions.structure_arrays import ARRAYS, HOST
from physics.engine.axiom_3_structure.logic.start_grid import start_grid
from physics.engine.axiom_3_structure.logic.relink import relink

wp.config.quiet = True

TEST_ARRAYS = {**ARRAYS, "quanta": ("int32", "box", False), "cell_quanta": ("int64", "cell", False),
               "pull": ("float32", "cell", False), "rate": ("float32", "cell", False),
               "way_turn": ("float32", "way", False)}


def small_grid(side: int = 4, spare: float = 0.25) -> Grid:
    """The start lattice on the CPU with the arrays the structure's logic reads."""
    return start_grid(side, 1, spare, TEST_ARRAYS, HOST, "cpu")


def snapshot(grid: Grid) -> dict:
    """Every array and table of the grid as host copies, to see what a function wrote."""
    held = {name: getattr(grid, name).numpy().copy() for name in grid.arrays}
    held.update({name: getattr(grid, name).copy() for name in ("fell_into", "torn_from", "made_kind")})
    held["adj_id"], held["adj_code"] = grid.adj_id.numpy().copy(), grid.adj_code.numpy().copy()
    held["counts"] = np.array([grid.count, grid.way_count, grid.way_passes, grid.version, grid.step])
    return held


def changed_names(before: dict, after: dict) -> set:
    """The names whose values differ between two snapshots (nan and inf compare as themselves)."""
    return {name for name in before
            if before[name].shape != after[name].shape or not np.array_equal(before[name], after[name], equal_nan=True)}


def tear_relations(grid: Grid, volume: int, axis: int) -> tuple:
    """The contract's relation rules of a tear of `volume` along `axis`, by hand; returns (new joint, new volume)."""
    new_joint, new_volume = grid.new_ids(2)
    step = [0, 0, 0]
    step[axis] = 1
    up, down = code_of_host(*step), opposite_host(code_of_host(*step))
    old, new, middle, touched = [], [], [], {volume, new_joint, new_volume}
    for joint, code in grid.neighbours(volume):
        side = step_of_host(code)[axis]
        theirs = grid.neighbours(joint)
        touched.add(joint)
        if side < 0:
            old.append((joint, code))
        elif side > 0:
            grid.set_neighbours(joint, [(new_volume, at) if (one, at) == (volume, opposite_host(code)) else (one, at)
                                        for one, at in theirs])
            new.append((joint, code))
        else:
            grid.set_neighbours(joint, theirs + [(new_volume, opposite_host(code)), (new_joint, opposite_host(code))])
            old.append((joint, code))
            new.append((joint, code))
            middle.append((joint, code))
    grid.set_neighbours(volume, old + [(new_joint, up)])
    grid.set_neighbours(new_volume, new + [(new_joint, down)])
    grid.set_neighbours(new_joint, middle + [(volume, down), (new_volume, up)])
    for cell, kind in ((new_joint, JOINT), (new_volume, ENERGY_VOLUME)):
        grid.set_cell("kind", cell, kind)
        grid.set_cell("made_kind", cell, kind)
        grid.set_cell("torn_from", cell, volume)
    relink(grid, sorted(touched))
    return new_joint, new_volume


@pytest.fixture
def grid() -> Grid:
    """The 4-a-side start lattice: 8 energy volumes wrapped in 56 joints."""
    return small_grid()
