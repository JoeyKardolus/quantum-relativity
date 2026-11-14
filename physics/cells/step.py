"""Step: one engine step; the rules run on the neighbour graph, then the shapes follow.

Contract: `step(engine, settings)` is pure: it returns a new Engine and leaves the
old one as it was. Deterministic: the same engine and settings give the same result.
  1. rules     grid = apply_rules(engine.grid, settings): pull, joint push, the
               leak, sizes from energy and joint collapse, exactly as on any
               grid, on the joints of the current tessellation (apply_rules.py).
               No rule reads positions, weights or shapes.
  2. seeds     positions relax toward their cell centroids and are drawn together
               across squeezed joints, each move at most MOVE_LIMIT (0.02) of its
               cell's width, so the shapes drift slowly and nothing jumps
               (move_seeds.py). Joint collapse removes seeds:
               of the energy volumes that fell into one, the one that held the most
               energy keeps its seed (lowest index on a tie), the others' seeds go;
               all their energy is already in it.
  3. weights   each kept seed's weight takes one Newton step toward its new target
               size (target_size.py, fit_weights.py).
  4. box       side = total_space() ** (1/3): pull shrinks space, push and the leak
               grow it, so the box follows; positions scale by the side ratio, weights by
               its square, which changes no shape.
  5. shapes    the new positions and weights are tessellated (tessellate.py): the
               shapes the viewer draws. The grid is woven: its joints, seams and joint
               volumes are what the rules left (apply_rules.py), so two energy volumes
               stop or start neighbouring only by joint collapse. Where the shapes
               change a neighbour pair by themselves (a power tessellation must, to give
               cells of very different sizes), the grid does not follow; the pairs are
               counted in neighbour_changes (neighbour_changes.py).
collapses grows by the energy volumes that fell in; grid.merged_into maps each old
energy volume to its new index.
"""
from __future__ import annotations

import numpy as np

from physics.cells.apply_rules import apply_rules
from physics.cells.engine import Engine
from physics.cells.engine_settings import EngineSettings
from physics.cells.fit_weights import fit_weights
from physics.cells.move_seeds import move_seeds
from physics.cells.neighbour_changes import neighbour_changes
from physics.cells.target_size import target_size
from physics.cells.tessellate import tessellate

MOVE_LIMIT = 0.02


def step(engine: Engine, settings: EngineSettings) -> Engine:
    """Run the rules on the woven grid, then move seeds and box and re-tessellate the shapes."""
    before = engine.grid
    ruled = apply_rules(before, settings)
    kept = _kept_seeds(ruled.merged_into, before.energy, ruled.cell_count)
    moved = move_seeds(engine.positions, engine.tessellation, before.joints, before.squeeze,
                       engine.side, MOVE_LIMIT)[kept]
    side = float(np.cbrt(ruled.total_space()))
    ratio = side / engine.side
    weights = fit_weights(engine.weights, engine.tessellation, kept,
                          target_size(ruled) / ratio ** 3) * ratio ** 2
    positions = np.mod(moved * ratio, side)
    tessellation = tessellate(positions, weights, side)
    changes = neighbour_changes(engine.tessellation.pairs, ruled.merged_into,
                                tessellation.pairs, ruled.joints)
    return Engine(grid=ruled, positions=positions, weights=weights, side=side,
                  tessellation=tessellation,
                  collapses=engine.collapses + before.cell_count - ruled.cell_count,
                  start_space=engine.start_space,
                  neighbour_changes=engine.neighbour_changes + changes)


def _kept_seeds(merged_into: np.ndarray, energy: np.ndarray, count: int) -> np.ndarray:
    """Per new energy volume, the old one whose seed it keeps: the most energy, then lowest index."""
    order = np.lexsort((np.arange(len(energy)), -energy, merged_into))
    first = np.ones(len(order), dtype=bool)
    first[1:] = merged_into[order][1:] != merged_into[order][:-1]
    kept = np.empty(count, dtype=np.int64)
    kept[merged_into[order][first]] = order[first]
    return kept
