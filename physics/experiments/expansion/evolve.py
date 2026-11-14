"""Evolve a clustered network under the rules and record what the network measures.

Contract: `evolve(network, labels, settings, steps, every, probes)` applies the
rules (`safe_step`) `steps` times and returns the record.
Every `every` steps it measures, from the network only: the scale factor (mean
length-weighted path distance over a fixed set of distant sparse-space joint
pairs, chosen at step 0), the mean sparse connection length, each cluster's
internal size (mean path distance from its probe joints to its own joints),
each cluster's joint count, the sparse joint count and the total joint count;
after every step it records the sparse joint count ("sparse_by_step").
Distances to a joint left with no connection (a whole region fell into one
joint, its energy held) are infinite and left out of the means. Probe joints
and labels follow joint collapse through
merged_into (`carried[original] = current index`; a merged joint takes the
highest label among its members, so a cluster swallowing sparse joints stays
the cluster). It stops early when the network can no longer be read
("breakdown_step" = that step, the last readable network measured), or when
any connection is shorter than COLLAPSE_LENGTH, which Dijkstra cannot resolve,
or when no connection is left (every joint fell into a few isolated joints).
"""
from __future__ import annotations

import numpy as np

from physics.experiments.follow import carry_joints, safe_step
from physics.measure import distances, mean_length
from physics.network import Network
from physics.rules import RuleSettings

COLLAPSE_LENGTH = 1e-9


def evolve(network: Network, labels: np.ndarray, settings: RuleSettings, steps: int,
           every: int, probes: dict):
    """Run the rules; return the measured histories."""
    carried = np.arange(network.joint_count)
    record = {"step": [], "scale": [], "sparse_length": [], "cluster_size": [], "cluster_joints": [],
              "sparse_joints": [], "joints": [], "sparse_by_step": [int(np.sum(labels == -1))],
              "breakdown_step": None, "breakdown_reason": None}
    for index in range(steps + 1):
        if index % every == 0:
            _measure(network, carried, labels, probes, record, index)
        if index == steps:
            break
        stepped, reason = safe_step(network, settings)
        if stepped is not None and stepped.connection_count == 0:
            stepped, reason = None, f"no connection left: everything fell into {stepped.joint_count} joint(s)"
        elif stepped is not None and stepped.length.min() < COLLAPSE_LENGTH:
            stepped, reason = None, f"a connection shorter than {COLLAPSE_LENGTH:g}"
        if stepped is None:
            record["breakdown_step"], record["breakdown_reason"] = index + 1, reason
            if index % every:
                _measure(network, carried, labels, probes, record, index)
            break
        network, carried = stepped, carry_joints(carried, stepped)
        record["sparse_by_step"].append(_sparse_count(network, carried, labels))
    return record


def _measure(network: Network, carried: np.ndarray, labels: np.ndarray, probes: dict,
             record: dict, index: int) -> None:
    current = _current_labels(network, carried, labels)
    sparse_sources = carried[probes["sparse_sources"]]
    cluster_sources = [carried[sources] for sources in probes["cluster_sources"]]
    table = distances(network, np.concatenate([sparse_sources, *cluster_sources]))
    sparse_connections = (current[network.pairs] == -1).all(axis=1)
    record["step"].append(index)
    record["scale"].append(_finite_mean(table[probes["pair_rows"], carried[probes["pair_targets"]]]))
    record["sparse_length"].append(mean_length(network, np.flatnonzero(sparse_connections)))
    record["joints"].append(network.joint_count)
    record["sparse_joints"].append(int(np.sum(current == -1)))
    record["cluster_joints"].append(np.bincount(current[current >= 0], minlength=len(cluster_sources)).tolist())
    offset, sizes = len(sparse_sources), []
    for cluster, sources in enumerate(cluster_sources):
        rows = table[offset: offset + len(sources)]
        sizes.append(_finite_mean(rows[:, current == cluster]))
        offset += len(sources)
    record["cluster_size"].append(sizes)


def _finite_mean(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(finite.mean()) if finite.size else float("nan")


def _current_labels(network: Network, carried: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Label of each current joint: the highest label among the original joints it holds."""
    current = np.full(network.joint_count, -1)
    np.maximum.at(current, carried, labels)
    return current


def _sparse_count(network: Network, carried: np.ndarray, labels: np.ndarray) -> int:
    return int(np.sum(_current_labels(network, carried, labels) == -1))
