"""Space dimension scene: the ball and the spectral dimension of the cell-contact graph, beside a plain grid's.

Contract: `space_dimension(cells_per_side=16, boxes_per_cell_side=4, steps=20, sources=8,
seed=1, settings=None, device="cuda")` builds new_universe(cells_per_side,
boxes_per_cell_side, seed) (the uneven start), measures it, takes `steps` steps
(default settings Settings.all_on()) and measures it again. A measurement reads only
which cells touch across a box face (physics.potts.contact_pairs on the owners), as an
unweighted graph of every living cell, energy volumes and joints alike (bible rule 1:
no coordinates), and runs physics.measure's ball_dimension and spectral_dimension from
the `sources` most central cells (central_nodes). The ranges follow the graph's own
size, the same rule as the measures' validation: with R a quarter of the first
source's farthest reachable distance in steps (at least 3), radii 2..R and walk times
max(2, R**2 // 10)..R**2. The same rule run on a plain cubic grid of cells_per_side
nodes a side (grid_graph) gives the reference: the number the estimator returns for
true dimension 3 at this size, since a small graph pulls both estimates off 3.
Returns a dict (no verdict):
  cells_per_side, boxes_per_cell_side, steps, seed, settings (dict), known_dimension (3)
  reference, start, end   each: ball_dimension, spectral_dimension, radii, times,
                          cells (nodes with a neighbour), shells, return_probability
"""
from __future__ import annotations

from dataclasses import asdict

import numpy as np
from scipy.sparse import csgraph

from physics.measure import adjacency, ball_dimension, central_nodes, grid_graph, spectral_dimension
from physics.potts import Settings, contact_pairs, new_universe, step

KNOWN_DIMENSION = 3


def space_dimension(cells_per_side: int = 16, boxes_per_cell_side: int = 4, steps: int = 20, sources: int = 8,
                    seed: int = 1, settings: Settings | None = None, device: str = "cuda") -> dict:
    """Ball and spectral dimension of the cell-contact graph at the start and after the steps."""
    settings = settings or Settings.all_on()
    universe = new_universe(cells_per_side, boxes_per_cell_side, seed=seed, device=device)
    start = _measured(_contact_matrix(universe.owner.numpy(), universe.cell_count), sources)
    for _ in range(steps):
        step(universe, settings)
    end = _measured(_contact_matrix(universe.owner.numpy(), universe.cell_count), sources)
    low, high, node_count = grid_graph(cells_per_side, KNOWN_DIMENSION)
    return {"cells_per_side": cells_per_side, "boxes_per_cell_side": boxes_per_cell_side, "steps": steps,
            "seed": seed, "settings": asdict(settings), "known_dimension": KNOWN_DIMENSION,
            "reference": _measured(adjacency(low, high, node_count), sources), "start": start, "end": end}


def _contact_matrix(owner: np.ndarray, cell_count: int):
    low, high, _ = contact_pairs(owner)
    return adjacency(low, high, cell_count)


def _measured(matrix, sources: int) -> dict:
    """Both dimensions from the most central nodes, ranges set by the graph's own reach."""
    picked = central_nodes(matrix, sources)
    distances = csgraph.shortest_path(matrix, unweighted=True, indices=picked[0])
    reach = max(3, int(distances[np.isfinite(distances)].max()) // 4)
    radii = np.arange(2, reach + 1)
    times = np.arange(max(2, reach ** 2 // 10), reach ** 2 + 1)
    ball = ball_dimension(matrix, picked, radii)
    spectral = spectral_dimension(matrix, picked, times)
    degree = np.asarray(matrix.sum(axis=1)).ravel()
    return {"ball_dimension": ball["dimension"], "spectral_dimension": spectral["dimension"],
            "radii": ball["radii"], "times": spectral["times"], "cells": int(np.sum(degree > 0)),
            "shells": ball["shells"], "return_probability": spectral["return_probability"]}
