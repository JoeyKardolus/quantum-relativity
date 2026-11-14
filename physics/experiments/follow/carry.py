"""Follow joints through joint collapse.

Contract: `carry_joints(carried, network)` takes `carried` (for each original
joint, its index in the previous network) and a network returned by a rules
step, and returns each original joint's index in that network, read from its
`merged_into` (unchanged when merged_into is None).
"""
from __future__ import annotations

import numpy as np

from physics.network import Network


def carry_joints(carried: np.ndarray, network: Network) -> np.ndarray:
    """Original joint -> index in the stepped network."""
    if network.merged_into is None:
        return carried
    return network.merged_into[carried]
