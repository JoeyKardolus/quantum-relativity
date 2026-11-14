"""First display layout of a network: landmark classical MDS into 3D. Display only.

Contract: `initial_layout(network, landmark_count)` picks landmark_count joints
by farthest-point sampling on network distance (starting from joint 0), runs
classical MDS on their distances and places every other joint by landmark
triangulation (de Silva and Tenenbaum, 2004). Returns an array
(joint_count, 3) for drawing; it is never stored on the network. Joints the
landmarks cannot reach sit at the origin.
"""
from __future__ import annotations

import numpy as np

from physics.measure import distances
from physics.network import Network


def initial_layout(network: Network, landmark_count: int = 48) -> np.ndarray:
    """3D display positions from network distances (landmark MDS)."""
    reach = _farthest_point_distances(network, min(landmark_count, network.joint_count))
    reach[~np.isfinite(reach)] = 0.0
    squared = reach ** 2
    landmarks = squared[:, _landmark_columns(reach)]
    centring = np.eye(len(landmarks)) - 1.0 / len(landmarks)
    values, vectors = np.linalg.eigh(-0.5 * centring @ landmarks @ centring)
    top = np.argsort(values)[::-1][:3]
    scale = np.sqrt(np.clip(values[top], 0.0, None))
    inverse = np.divide(vectors[:, top], scale, out=np.zeros_like(vectors[:, top]), where=scale > 0)
    return -0.5 * (squared - landmarks.mean(axis=1, keepdims=True)).T @ inverse


def _farthest_point_distances(network: Network, count: int) -> np.ndarray:
    rows = [distances(network, [0])[0]]
    nearest = rows[0].copy()
    for _ in range(count - 1):
        nearest[~np.isfinite(nearest)] = -1.0
        rows.append(distances(network, [int(np.argmax(nearest))])[0])
        nearest = np.minimum(nearest, rows[-1])
    return np.array(rows)


def _landmark_columns(reach: np.ndarray) -> np.ndarray:
    """Each landmark row's own joint is where its distance is zero."""
    return np.argmin(reach, axis=1)
