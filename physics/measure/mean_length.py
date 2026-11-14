"""Mean connection length: the network's scale factor, read directly.

Contract: `mean_length(network, connections)` is the mean length over the given
connections (default all). Tracked over steps it is the expansion history.
"""
from __future__ import annotations

import numpy as np

from physics.network import Network


def mean_length(network: Network, connections: np.ndarray | None = None) -> float:
    """Average length of the chosen connections."""
    lengths = network.length if connections is None else network.length[connections]
    return float(np.mean(lengths))
