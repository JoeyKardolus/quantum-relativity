"""Lump tracking read from the network only, against a ruler of the far background.

Contract: `track_lumps(network, joints, far_joints, candidates)` measures, on
the current network:
  - "background_length": mean length of the connections with both ends in
    `far_joints` (joints far from both lumps), the ruler that separates a lump
    approach from a whole-space shrink or growth;
  - "joint_distance": network distance between the two starting joints;
  - per lump, its centre: among the `candidates` joints with the most excess
    energy density on that lump's side (closer to its own starting joint),
    the one minimising the excess-weighted sum of squared network distances;
    its "width" (excess-weighted RMS distance from that centre) and its
    "peak" (largest excess density on its side);
  - "centre_distance": network distance between the two centres.
Excess density = mean connection energy at a joint minus the median of that
over `far_joints` (the background energy now, so drift or a squeeze of the
whole network does not count as excess).
"""
from __future__ import annotations

import numpy as np

from physics.measure import distances
from physics.network import Network


def track_lumps(network: Network, joints: tuple[int, int], far_joints: np.ndarray,
                candidates: int = 12) -> dict:
    """Background ruler, distances, centres, widths and peaks of two lumps on the current network."""
    from_joints = distances(network, list(joints))
    density = _joint_density(network)
    excess = np.clip(density - float(np.median(density[far_joints])), 0.0, None)
    sides = [from_joints[0] < from_joints[1], from_joints[1] <= from_joints[0]]
    lumps = [_lump(network, excess, side, candidates) for side in sides]
    centre_distance = float(distances(network, [lumps[0]["centre"]])[0, lumps[1]["centre"]])
    return {"background_length": _background_length(network, far_joints),
            "joint_distance": float(from_joints[0, joints[1]]),
            "centre_distance": centre_distance, "lumps": lumps}


def _background_length(network: Network, far_joints: np.ndarray) -> float:
    far = np.zeros(network.joint_count, dtype=bool)
    far[far_joints] = True
    inside = far[network.pairs[:, 0]] & far[network.pairs[:, 1]]
    return float(network.length[inside].mean()) if inside.any() else float("nan")


def _joint_density(network: Network) -> np.ndarray:
    sums = np.bincount(network.pairs.ravel(), weights=np.repeat(network.energy, 2),
                       minlength=network.joint_count)
    return sums / np.maximum(network.degree(), 1)


def _lump(network: Network, excess: np.ndarray, side: np.ndarray, candidates: int) -> dict:
    weights = np.where(side, excess, 0.0)
    order = np.argsort(weights)[::-1][:candidates]
    if weights[order[0]] <= 0:
        return {"centre": int(order[0]), "width": float("nan"), "peak": 0.0}
    from_candidates = distances(network, order)
    finite = np.where(np.isfinite(from_candidates), from_candidates, 0.0)
    spread = (finite ** 2) @ weights
    best = int(np.argmin(spread))
    width = float(np.sqrt(spread[best] / weights.sum()))
    return {"centre": int(order[best]), "width": width, "peak": float(weights[order[0]])}
