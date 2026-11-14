"""The even version of a scene: the same joints and connections with its energy spread evenly.

Contract: `even_version(network)` returns a Network with the same joints and
connections in which every connection holds the scene's mean connection energy
(so the total energy is the same) and the energy-length length. Energy flow
drives a scene toward this state; it is the even space the balancing drift is
calibrated on, the way Einstein balanced the mean density of a static universe.
On it every joint has the same pull, (mean energy)^2 (pull.py: source E/V =
energy^2, and an even pull solves the grounded Poisson equation exactly).
"""
from __future__ import annotations

import numpy as np

from physics.network import LENGTH_SCALE, Network


def even_version(network: Network) -> Network:
    """Same connections, every one holding the mean energy."""
    mean_energy = float(network.energy.mean())
    return Network.from_lengths(network.joint_count, network.pairs,
                                np.full(network.connection_count, LENGTH_SCALE / mean_energy))
