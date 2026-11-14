"""New engine: a periodic box of energy volumes with slightly uneven energy, shapes settled.

Contract: `new_engine(cells, seed, unevenness)` returns an Engine (engine.py) of
exactly `cells` energy volumes:
  energy     1 + unevenness * u, u uniform in [-1, 1] per energy volume (seeded)
  volume     rest_volume(energy): the energy-volume rule
  positions  uniform at random in the box (seeded); weights 0 (plain Voronoi)
  joints     the shared faces of their tessellation, each at its rest volume
             (joint_rest.py, the EngineSettings.all_on() share), squeeze 0
  side       total_space() ** (1/3) (a plain Voronoi tessellation does not
             depend on the box's scale, so the faces are found in a unit box first)
  settle     SETTLE_ROUNDS (12) rounds of the shapes part of step() with no rule:
             seeds relax, weights fit the target sizes, re-tessellate, reconcile
             joints. Energy and energy volumes do not change; the joints become
             the faces of the settled shapes. Nothing is placed by hand.
collapses 0, start_space the settled total_space(). Raises ValueError when
unevenness is outside [0, 1) (energy must stay positive) or cells < 30.
"""
from __future__ import annotations

import numpy as np

from physics.cells.engine import Engine
from physics.cells.engine_settings import EngineSettings
from physics.cells.fit_weights import fit_weights
from physics.cells.grid import Grid
from physics.cells.joint_rest import joint_rest
from physics.cells.move_seeds import move_seeds
from physics.cells.reconcile_joints import reconcile_joints
from physics.cells.rest_volume import rest_volume
from physics.cells.target_size import target_size
from physics.cells.tessellate import tessellate

SETTLE_ROUNDS = 12


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
    pairs = tessellate(unit, weights, 1.0).pairs
    grid = Grid(cell_count=cells, joints=pairs, energy=energy, volume=volume,
                joint_volume=joint_rest(volume, pairs, settings.joint_rest),
                squeeze=np.zeros(len(pairs)), charge=np.zeros(cells))
    side = float(np.cbrt(grid.total_space()))
    positions = unit * side
    tessellation = tessellate(positions, weights, side)
    everyone = np.arange(cells)
    for _ in range(SETTLE_ROUNDS):
        positions = move_seeds(positions, tessellation, grid.joints, grid.squeeze, side)
        weights = fit_weights(weights, tessellation, everyone, target_size(grid))
        tessellation = tessellate(positions, weights, side)
        grid = reconcile_joints(grid, tessellation.pairs, settings)
    return Engine(grid=grid, positions=positions, weights=weights, side=side,
                  tessellation=tessellation, collapses=0, start_space=grid.total_space())

