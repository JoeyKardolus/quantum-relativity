"""Causal reference network: a causal set sprinkled into a flat causal diamond.

Contract: `causal_reference(joint_count, dimension, proper_time, anchors, seed)`
builds the diamond between the events (t=0, x=0) and (t=proper_time, x=0) in
1+1 (dimension=2) or 3+1 (dimension=4) flat spacetime. Joint 0 is the bottom
tip, joint 1 the top tip, joints 2.. are the optional `anchors` (an array of
(t, x...) events inside the diamond, for twin-paradox style paths), then
joint_count points are scattered uniformly (Poisson-style fixed count). The
causal relation is "inside the other's future light cone"; connections are the
links (relations with nothing between), each given energy 1 because a link has
no meaningful spatial length. All coordinates are DISCARDED. Returns the
Network and the known answers {"dimension", "density", "volume",
"proper_time", "bottom", "top", "anchors"} (joint indices, no coordinates).
"""
from __future__ import annotations

import math

import numpy as np
import scipy.sparse as sp

from physics.network.network import Network
from physics.network.energy_length import energy_length


def causal_reference(joint_count: int, dimension: int = 2, proper_time: float = 1.0,
                     anchors: np.ndarray | None = None, seed: int = 0) -> tuple[Network, dict]:
    """A sprinkled causal diamond with its order and links; coordinates are not kept."""
    if dimension not in (2, 4):
        raise ValueError("dimension must be 2 (1+1) or 4 (3+1)")
    rng = np.random.default_rng(seed)
    tips = np.zeros((2, dimension))
    tips[1, 0] = proper_time
    extra = np.zeros((0, dimension)) if anchors is None else np.atleast_2d(anchors).astype(float)
    events = np.vstack([tips, extra, _sprinkle(rng, joint_count, dimension, proper_time)])
    relation = _relation(events)
    links = _links(relation)
    heads, tails = np.nonzero(links)
    pairs = np.stack([heads, tails], axis=1).astype(np.int64)
    energies = np.ones(len(pairs))
    network = Network(len(events), pairs, energies, energy_length(energies),
                      causal=sp.csr_matrix(relation))
    volume = _diamond_volume(dimension, proper_time)
    del events
    known = {"dimension": dimension, "density": joint_count / volume, "volume": volume,
             "proper_time": proper_time, "bottom": 0, "top": 1,
             "anchors": list(range(2, 2 + len(extra)))}
    return network, known


def _diamond_volume(dimension: int, proper_time: float) -> float:
    if dimension == 2:
        return proper_time ** 2 / 2
    return math.pi * proper_time ** 4 / 24


def _sprinkle(rng: np.random.Generator, count: int, dimension: int,
              proper_time: float) -> np.ndarray:
    if dimension == 2:
        light_u, light_v = rng.random(count) * proper_time, rng.random(count) * proper_time
        return np.stack([(light_u + light_v) / 2, (light_u - light_v) / 2], axis=1)
    kept: list[np.ndarray] = []
    total = 0
    half = proper_time / 2
    while total < count:
        box = rng.random((4 * count, 4)) * [proper_time, 2 * half, 2 * half, 2 * half]
        box[:, 1:] -= half
        radius = np.linalg.norm(box[:, 1:], axis=1)
        inside = box[(radius < box[:, 0]) & (radius < proper_time - box[:, 0])]
        kept.append(inside)
        total += len(inside)
    return np.vstack(kept)[:count]


def _relation(events: np.ndarray) -> np.ndarray:
    """relation[i, j] = event j lies strictly inside event i's future light cone."""
    times = events[:, 0]
    space = events[:, 1:]
    relation = np.zeros((len(events), len(events)), dtype=bool)
    for start in range(0, len(events), 1024):
        stop = min(start + 1024, len(events))
        gap = times[None, :] - times[start:stop, None]
        apart = np.linalg.norm(space[None, :, :] - space[start:stop, None, :], axis=2)
        relation[start:stop] = gap > apart
    return relation


def _links(relation: np.ndarray) -> np.ndarray:
    """Relations with no joint in between (the Hasse diagram)."""
    as_float = relation.astype(np.float32)
    between = as_float @ as_float
    return relation & (between == 0)
