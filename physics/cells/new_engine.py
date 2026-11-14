"""New engine: a periodic box of energy volumes with slightly uneven energy, shapes settled.

Contract: `new_engine(cells, seed, unevenness)` returns an Engine (engine.py) of
exactly `cells` energy volumes:
  energy     1 + unevenness * u, u uniform in [-1, 1] per energy volume (seeded)
  volume     rest_volume(energy): the energy-volume rule
  positions  uniform at random in the box (seeded); weights 0 (plain Voronoi)
  joints     the shared faces of their tessellation, each at its balance
             (joint_balance.py, the EngineSettings.all_on() settings): pull and push
             cancel, so its squeeze stays below the seal point and far below the push
             limit and no joint collapses in step 1; seams its edges
  side       total_space() ** (1/3) (a plain Voronoi tessellation does not
             depend on the box's scale, so the faces are found in a unit box first)
  settle     SETTLE_ROUNDS (40) rounds of: seeds relax uncapped (move_seeds.py),
             weights fit the target sizes, re-tessellate, reconcile joints, joints back
             to their balance; then FIT_ROUNDS (10) rounds of the same with the seeds
             held, so each cell's size matches its energy before step 1. Energy and
             energy volumes do not change; the joints become the faces of the settled
             shapes. Last the box takes side total_space() ** (1/3) again (positions
             times the side ratio, weights times its square: the same shapes).
             Nothing is placed by hand.
collapses 0, start_space the settled total_space(). Raises ValueError when
unevenness is outside [0, 1) (energy must stay positive) or cells < 30.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from physics.cells.engine import Engine
from physics.cells.engine_settings import EngineSettings
from physics.cells.fit_weights import fit_weights
from physics.cells.grid import Grid
from physics.cells.joint_balance import joint_balance
from physics.cells.joint_rest import joint_rest
from physics.cells.joint_squeeze import joint_squeeze
from physics.cells.move_seeds import move_seeds
from physics.cells.reconcile_joints import reconcile_joints
from physics.cells.rest_volume import rest_volume
from physics.cells.seam_openness import seam_openness
from physics.cells.target_size import target_size
from physics.cells.tessellate import tessellate

SETTLE_ROUNDS = 40
FIT_ROUNDS = 10


def new_engine(cells: int, seed: int, unevenness: float) -> Engine:
    """An engine of `cells` energy volumes with slightly uneven energy and settled shapes."""
    if not 0 <= unevenness < 1:
        raise ValueError("unevenness must be in [0, 1) so every energy stays positive")
    if cells < 30:
        raise ValueError("ask for at least 30 cells so the periodic box is sound")
    settings = EngineSettings.all_on()
    rng = np.random.default_rng(seed)
    energy = 1.0 + unevenness * rng.uniform(-1.0, 1.0, cells)
    volume = rest_volume(energy)
    unit = rng.random((cells, 3))
    weights = np.zeros(cells)
    first = tessellate(unit, weights, 1.0)
    pairs = first.pairs
    grid = Grid(cell_count=cells, joints=pairs, energy=energy, volume=volume,
                joint_volume=joint_rest(volume, pairs, settings.joint_rest),
                squeeze=np.zeros(len(pairs)), charge=np.zeros(cells), seams=first.seams,
                seam_open=np.ones(len(first.seams)))
    grid = _balanced(grid, settings)
    side = float(np.cbrt(grid.total_space()))
    positions = unit * side
    tessellation = tessellate(positions, weights, side)
    everyone = np.arange(cells)
    for round_number in range(SETTLE_ROUNDS + FIT_ROUNDS):
        if round_number < SETTLE_ROUNDS:
            positions = move_seeds(positions, tessellation, grid.joints, grid.squeeze, side,
                                   np.inf)
        weights = fit_weights(weights, tessellation, everyone, target_size(grid))
        tessellation = tessellate(positions, weights, side)
        grid = _balanced(reconcile_joints(grid, tessellation.pairs, tessellation.seams,
                                          settings), settings)
    ratio = float(np.cbrt(grid.total_space())) / side
    side, positions, weights = side * ratio, positions * ratio, weights * ratio ** 2
    tessellation = tessellate(positions, weights, side)
    grid = reconcile_joints(grid, tessellation.pairs, tessellation.seams, settings)
    return Engine(grid=grid, positions=positions, weights=weights, side=side,
                  tessellation=tessellation, collapses=0, start_space=grid.total_space(),
                  neighbour_changes=0)


def _balanced(grid: Grid, settings: EngineSettings) -> Grid:
    """The grid with every joint at its balance volume, squeeze and seams to match."""
    balanced = replace(grid, joint_volume=joint_balance(grid, settings))
    return replace(balanced, squeeze=joint_squeeze(balanced, settings),
                   seam_open=seam_openness(balanced, settings))
