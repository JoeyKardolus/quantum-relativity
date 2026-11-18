"""Volume graph: which energy volumes of the engine's grid neighbour which, as a plain graph.

Contract: `volume_graph(ways, kinds)` takes the engine's living ways
(`physics.engine.read_ways(state)`: int rows `(slot, a, b, joint)`) and every cell's kind
(`physics.engine.read_cells(state)["kind"]`) and returns `(matrix, volumes)`:
  volumes  the ids of the living energy volumes, rising (int64); node n of the graph is
           volume volumes[n]
  matrix   the symmetric 0/1 adjacency (definitions/adjacency.py) of those nodes: two
           volumes touch when at least one way goes between them through a joint.
           Several ways between the same two volumes count once
A way whose two ends are the same volume (a joint all of whose volumes have merged
into one) joins nothing and is left out. Only relations are read: no length, no energy,
no box (axiom 3: a cell has only relations and a size). Host arrays only; nothing in the
engine is changed. Raises ValueError when a way ends on a cell that is not a living
energy volume, or when there is no energy volume.
"""
from __future__ import annotations

import numpy as np
from scipy import sparse

from physics.engine import ENERGY_VOLUME
from physics.measure.dimension.definitions.adjacency import adjacency

_FIRST_VOLUME, _SECOND_VOLUME = 1, 2


def volume_graph(ways: np.ndarray, kinds: np.ndarray) -> tuple[sparse.csr_matrix, np.ndarray]:
    """The energy volumes' neighbour graph and the volume id of each of its nodes."""
    kinds = np.asarray(kinds)
    ways = np.asarray(ways, dtype=np.int64).reshape(-1, 4)
    volumes = np.flatnonzero(kinds == ENERGY_VOLUME).astype(np.int64)
    if volumes.size == 0:
        raise ValueError("there is no energy volume")
    first, second = ways[:, _FIRST_VOLUME], ways[:, _SECOND_VOLUME]
    ends = np.concatenate([first, second])
    if ends.size and (ends.min() < 0 or ends.max() >= kinds.size or np.any(kinds[ends] != ENERGY_VOLUME)):
        raise ValueError("a way ends on a cell that is not a living energy volume")
    node = np.full(kinds.size, -1, dtype=np.int64)
    node[volumes] = np.arange(volumes.size)
    apart = first != second
    return adjacency(node[first[apart]], node[second[apart]], int(volumes.size)), volumes
