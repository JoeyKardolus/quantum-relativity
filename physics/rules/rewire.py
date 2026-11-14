"""Rewiring: local slides of one connection end, kept only when they lower the action.

Contract: `rewire(network, settings, rng)` makes `settings.rewiring` attempts.
An attempt picks a random connection (a, b) and a random other neighbour c of
b, and proposes to slide the b end to c, giving (a, c): a strictly local move
(c is two steps from a). It is refused if c is a or already joined to a. It is
accepted only if it strictly lowers the action
  S = sum_j (degree_j - target_degree)^2
      + triangle_weight * sum_j (triangles_j - target_triangles)^2,
where triangles_j counts triangles through joint j. The moved connection keeps
its energy and length. Connection count and total energy never change.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import replace

import numpy as np

from physics.network import Network


def rewire(network: Network, settings, rng: np.random.Generator) -> Network:
    """Up to settings.rewiring accepted local slides that each lower the action."""
    pairs = network.pairs.copy()
    neighbours = [set() for _ in range(network.joint_count)]
    for head, tail in pairs:
        neighbours[head].add(tail)
        neighbours[tail].add(head)
    degree = network.degree().astype(float)
    triangles = _triangles(neighbours)
    for _ in range(settings.rewiring):
        index = int(rng.integers(len(pairs)))
        head, tail = (int(joint) for joint in pairs[index][rng.permutation(2)])
        options = list(neighbours[tail] - {head})
        if not options:
            continue
        target = options[int(rng.integers(len(options)))]
        if target in neighbours[head]:
            continue
        change = _triangle_change(neighbours, head, tail, target)
        if _action_change(degree, triangles, tail, target, change, settings) >= 0:
            continue
        neighbours[head].discard(tail)
        neighbours[tail].discard(head)
        neighbours[head].add(target)
        neighbours[target].add(head)
        degree[tail] -= 1
        degree[target] += 1
        for joint, delta in change.items():
            triangles[joint] += delta
        pairs[index] = (min(head, target), max(head, target))
    return replace(network, pairs=pairs)


def _triangles(neighbours: list[set]) -> np.ndarray:
    counts = np.zeros(len(neighbours))
    for joint, around in enumerate(neighbours):
        counts[joint] = sum(len(around & neighbours[other]) for other in around) / 2
    return counts


def _triangle_change(neighbours: list[set], head: int, tail: int, target: int) -> Counter:
    """Per-joint triangle change from replacing connection (head, tail) by (head, target)."""
    change: Counter = Counter()
    for shared in neighbours[head] & neighbours[tail]:
        for joint in (head, tail, shared):
            change[joint] -= 1
    around_head = neighbours[head] - {tail}
    for shared in around_head & neighbours[target]:
        for joint in (head, target, shared):
            change[joint] += 1
    return change


def _action_change(degree: np.ndarray, triangles: np.ndarray, tail: int, target: int,
                   change: Counter, settings) -> float:
    goal = settings.target_degree
    degree_term = ((degree[tail] - 1 - goal) ** 2 - (degree[tail] - goal) ** 2
                   + (degree[target] + 1 - goal) ** 2 - (degree[target] - goal) ** 2)
    aim = settings.target_triangles
    triangle_term = sum((triangles[joint] + delta - aim) ** 2 - (triangles[joint] - aim) ** 2
                        for joint, delta in change.items())
    return degree_term + settings.triangle_weight * triangle_term
