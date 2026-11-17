"""Space dimension scene: the ball and the spectral dimension of the volume-neighbour graph, beside a plain grid's.

Contract: `space_dimension(cells_per_side=16, steps=20, sources=8, seed=1, settings=None,
device="cuda", start=None)` builds new_universe(cells_per_side, seed, **start) (the uneven start), measures
it, takes `steps` steps (default settings Settings(), every rule on) and measures it
again. A measurement reads only which energy volumes neighbour which, as an unweighted
graph (bible rule 1: no coordinates): two volumes neighbour when a joint carries a way
between them (physics.engine way_list). It runs physics.measure's ball_dimension and
spectral_dimension from the `sources` most central volumes (central_nodes). The ranges
follow the graph's own size, the same rule as the measures' validation: with R a
quarter of the first source's farthest reachable distance in steps (at least 3), radii
2..R and walk times max(2, R**2 // 10)..R**2. The same rule run on a plain cubic grid
of as many nodes a side as the start has energy volumes a side, closed round on itself
as the universe is, gives the reference: the number the estimator returns for true
dimension 3 at this size and shape, since a small graph pulls both estimates off 3.
Returns a dict (no verdict):
  cells_per_side, steps, seed, settings (dict), known_dimension (3)
  reference, start, end   each: ball_dimension, spectral_dimension, radii, times,
                          cells (volumes with a neighbour), shells, return_probability
"""
from __future__ import annotations

from dataclasses import asdict

import numpy as np
from scipy.sparse import csgraph

from physics.engine import ENERGY_VOLUME, Settings, State, new_universe, step, way_list
from physics.measure import adjacency, ball_dimension, central_nodes, grid_graph, spectral_dimension

KNOWN_DIMENSION = 3


def space_dimension(cells_per_side: int = 16, steps: int = 20, sources: int = 8, seed: int = 1,
                    settings: Settings | None = None, device: str = "cuda", start: dict | None = None) -> dict:
    """Ball and spectral dimension of the volume-neighbour graph at the start and after the steps."""
    settings = settings or Settings()
    universe = new_universe(cells_per_side, seed=seed, device=device, **(start or {}))
    start = _measured(_neighbour_matrix(universe), sources)
    for _ in range(steps):
        step(universe, settings)
    end = _measured(_neighbour_matrix(universe), sources)
    low, high, node_count = _plain_grid(max(round(start["cells"] ** (1.0 / 3.0)), 2), True)
    return {"cells_per_side": cells_per_side, "steps": steps, "seed": seed, "settings": asdict(settings), "known_dimension": KNOWN_DIMENSION,
            "reference": _measured(adjacency(low, high, node_count), sources), "start": start, "end": end}


def _plain_grid(side: int, wrap: bool) -> tuple[np.ndarray, np.ndarray, int]:
    """A plain cubic grid's neighbouring pairs, closed round on itself when the universe wraps."""
    low, high, node_count = grid_graph(side, KNOWN_DIMENSION)
    if not wrap or side < 3:
        return low, high, node_count
    ids = np.arange(node_count, dtype=np.int64).reshape((side,) * KNOWN_DIMENSION)
    first = [np.moveaxis(ids, axis, 0)[0].ravel() for axis in range(KNOWN_DIMENSION)]
    last = [np.moveaxis(ids, axis, 0)[-1].ravel() for axis in range(KNOWN_DIMENSION)]
    return np.concatenate([low, *first]), np.concatenate([high, *last]), node_count


def _neighbour_matrix(universe: State):
    """Energy volumes linked when a joint carries a way between them."""
    _, first, second = way_list(universe)
    volumes = np.flatnonzero(universe.kind_h[:universe.count] == ENERGY_VOLUME)
    number = np.full(universe.count, -1, dtype=np.int64)
    number[volumes] = np.arange(len(volumes))
    return adjacency(number[first], number[second], max(len(volumes), 1))


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
