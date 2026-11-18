"""Shared by the tests on the real engine: the device, and the reference the start must equal."""
from __future__ import annotations

import pytest
import warp as wp

from physics.measure import adjacency, grid_graph

wp.config.quiet = True


@pytest.fixture(scope="session")
def device() -> str:
    """The GPU when there is one, else the CPU: the measures read host arrays either way."""
    return "cuda" if wp.is_cuda_available() else "cpu"


@pytest.fixture(scope="session")
def lattice_graph():
    """The wrapped 26-neighbour grid of `volumes_per_side` cubed nodes: the engine's start as a graph."""
    def _graph(volumes_per_side: int):
        return adjacency(*grid_graph(volumes_per_side, 3, wrapped=True, diagonals=True))
    return _graph
