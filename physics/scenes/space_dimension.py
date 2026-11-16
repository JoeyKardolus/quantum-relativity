"""Space dimension scene: the ball and the spectral dimension of the cell-contact graph, beside a plain grid's.

Contract: `space_dimension(cells_per_side=16, boxes_per_cell_side=4, steps=20, sources=8,
seed=1, settings=None, device="cuda")` builds new_universe(cells_per_side,
boxes_per_cell_side, seed) (the uneven start), measures it, takes `steps` steps
(default settings Settings(), every rule on) and measures it again. A measurement reads
only which energy volumes neighbour which across the joint web, as an unweighted graph
(bible rule 1: no coordinates): the engine keeps no joint labels and its joints form one
web, so every occupied box is given to the energy volume whose plus box lies nearest it
(physics.engine cells; Euclidean distance, scipy distance_transform_edt), and two
volumes neighbour when their shares touch across a box face (physics.measure
contact_pairs). It runs physics.measure's ball_dimension and spectral_dimension from the
`sources` most central volumes (central_nodes). The ranges follow the graph's own
size, the same rule as the measures' validation: with R a quarter of the first
source's farthest reachable distance in steps (at least 3), radii 2..R and walk times
max(2, R**2 // 10)..R**2. The same rule run on a plain cubic grid of as many nodes a side
as the start has energy volumes a side ((cells_per_side - 1) // 2, grid_graph) gives the
reference: the number the estimator returns for
true dimension 3 at this size, since a small graph pulls both estimates off 3.
Returns a dict (no verdict):
  cells_per_side, boxes_per_cell_side, steps, seed, settings (dict), known_dimension (3)
  reference, start, end   each: ball_dimension, spectral_dimension, radii, times,
                          cells (volumes with a neighbour), shells, return_probability
"""
from __future__ import annotations

from dataclasses import asdict

import numpy as np
from scipy import ndimage
from scipy.sparse import csgraph

from physics.engine import Settings, State, cells, host_boxes, new_universe, step
from physics.measure import adjacency, ball_dimension, central_nodes, contact_pairs, grid_graph, spectral_dimension

KNOWN_DIMENSION = 3


def space_dimension(cells_per_side: int = 16, boxes_per_cell_side: int = 4, steps: int = 20, sources: int = 8,
                    seed: int = 1, settings: Settings | None = None, device: str = "cuda") -> dict:
    """Ball and spectral dimension of the cell-contact graph at the start and after the steps."""
    settings = settings or Settings()
    universe = new_universe(cells_per_side, boxes_per_cell_side, seed=seed, device=device)
    start = _measured(_contact_matrix(universe), sources)
    for _ in range(steps):
        step(universe, settings)
    end = _measured(_contact_matrix(universe), sources)
    low, high, node_count = grid_graph(max((cells_per_side - 1) // 2, 2), KNOWN_DIMENSION)
    return {"cells_per_side": cells_per_side, "boxes_per_cell_side": boxes_per_cell_side, "steps": steps,
            "seed": seed, "settings": asdict(settings), "known_dimension": KNOWN_DIMENSION,
            "reference": _measured(adjacency(low, high, node_count), sources), "start": start, "end": end}


def _contact_matrix(universe: State):
    """Energy volumes linked when the occupied boxes nearest each touch across a face."""
    found = cells(host_boxes(universe))
    owner = found["owner"]
    volumes = found["volumes"]
    plus = (owner >= 0) & (owner < volumes)
    _, nearest = ndimage.distance_transform_edt(~plus, return_indices=True)
    share = owner[tuple(nearest)]
    share = np.where((owner >= 0) & plus.any(), share, -1)
    low, high, _ = contact_pairs(share)
    return adjacency(low, high, max(volumes, 1))


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
