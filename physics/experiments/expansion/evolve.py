"""Evolve a clustered network under drift + attraction and record what the network measures.

Contract: `evolve(network, labels, settings, steps, every, probes)` applies
`physics.rules.step` `steps` times. Every `every` steps it measures, from the
network only: the scale factor (mean length-weighted path distance over a fixed
set of distant sparse-space joint pairs, chosen at step 0), the mean sparse
connection length, and each cluster's internal size (mean path distance between
its own joints). It stops early, and says so, when any connection collapses
(the last network before the collapse is measured)
(length below COLLAPSE_LENGTH or a non-finite energy): past that the rules
produce zero lengths and infinite energies, which Dijkstra cannot read.
"""
from __future__ import annotations

import numpy as np

from physics.measure import distances, mean_length
from physics.network import Network
from physics.rules import RuleSettings, step

COLLAPSE_LENGTH = 1e-9


def evolve(network: Network, labels: np.ndarray, settings: RuleSettings, steps: int,
           every: int, probes: dict) -> dict:
    """Run the rules; return the measured histories and the collapse step (or None)."""
    sparse_connections = np.flatnonzero((labels[network.pairs] == -1).all(axis=1))
    record = {"step": [], "scale": [], "sparse_length": [], "cluster_size": [], "collapse_step": None}
    for index in range(steps + 1):
        if index % every == 0:
            _measure(network, sparse_connections, probes, labels, record, index)
        if index == steps:
            break
        stepped = step(network, settings)
        if _collapsed(stepped):
            record["collapse_step"] = index + 1
            if index % every:
                _measure(network, sparse_connections, probes, labels, record, index)
            break
        network = stepped
    return record


def _collapsed(network: Network) -> bool:
    return bool(network.length.min() < COLLAPSE_LENGTH or not np.isfinite(network.energy).all())


def _measure(network: Network, sparse_connections: np.ndarray, probes: dict,
             labels: np.ndarray, record: dict, index: int) -> None:
    sources = np.concatenate([probes["sparse_sources"], *probes["cluster_sources"]])
    table = distances(network, sources)
    sparse_rows = table[: len(probes["sparse_sources"])]
    record["step"].append(index)
    record["scale"].append(float(sparse_rows[probes["pair_rows"], probes["pair_targets"]].mean()))
    record["sparse_length"].append(mean_length(network, sparse_connections))
    offset, sizes = len(probes["sparse_sources"]), []
    for cluster, cluster_sources in enumerate(probes["cluster_sources"]):
        rows = table[offset: offset + len(cluster_sources)]
        sizes.append(float(rows[:, labels == cluster].mean()))
        offset += len(cluster_sources)
    record["cluster_size"].append(sizes)
