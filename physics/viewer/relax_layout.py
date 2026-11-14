"""Move a display layout toward the network's current lengths: sparse stress majorization.

Contract: `relax_layout(network, positions, rng, iterations)` returns new
(joint_count, 3) display positions after `iterations` localized stress
majorization updates (Jacobi SMACOF): each joint moves to the weighted mean of
where each of its terms wants it. Terms are every connection (target its
length) plus a few far pairs (FAR_SOURCES random joints to FAR_TARGETS random
joints each, target their network distance), weight 1 / target^2. Targets are
floored at LENGTH_FLOOR times the median length so a near-zero connection
cannot dominate. Reads the network, never writes to it.
"""
from __future__ import annotations

import numpy as np

from physics.measure import distances
from physics.network import Network

FAR_SOURCES = 4
FAR_TARGETS = 150
LENGTH_FLOOR = 1e-3


def relax_layout(network: Network, positions: np.ndarray, rng: np.random.Generator,
                 iterations: int) -> np.ndarray:
    """Positions after a few stress-majorization sweeps toward current lengths."""
    heads, tails, target = _terms(network, rng)
    weight = 1.0 / target ** 2
    total = np.bincount(np.concatenate([heads, tails]), np.concatenate([weight, weight]),
                        network.joint_count)
    held = total > 0
    for _ in range(iterations):
        offset = positions[heads] - positions[tails]
        gap = np.linalg.norm(offset, axis=1, keepdims=True)
        reach = np.divide(target[:, None] * offset, gap, out=np.zeros_like(offset), where=gap > 0)
        wanted = np.zeros_like(positions)
        np.add.at(wanted, heads, weight[:, None] * (positions[tails] + reach))
        np.add.at(wanted, tails, weight[:, None] * (positions[heads] - reach))
        positions = np.where(held[:, None], wanted / np.maximum(total, 1e-300)[:, None], positions)
    return positions


def _terms(network: Network, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sources = rng.choice(network.joint_count, min(FAR_SOURCES, network.joint_count), replace=False)
    ends = rng.choice(network.joint_count, (len(sources), min(FAR_TARGETS, network.joint_count)))
    reach = distances(network, sources)[np.arange(len(sources))[:, None], ends]
    far_heads = np.repeat(sources, ends.shape[1])
    usable = np.isfinite(reach.ravel()) & (far_heads != ends.ravel())
    heads = np.concatenate([network.pairs[:, 0], far_heads[usable]])
    tails = np.concatenate([network.pairs[:, 1], ends.ravel()[usable]])
    floor = LENGTH_FLOOR * float(np.median(network.length))
    target = np.maximum(np.concatenate([network.length, reach.ravel()[usable]]), floor)
    return heads, tails, target
