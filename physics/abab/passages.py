"""Passages: every way energy can go from one energy volume through one joint to another (A -> B -> A).

Contract: `passages(contacts)` returns (giver, joint, taker), three int64 arrays of
equal length, one row per ordered pair of different energy volumes that touch the
same joint: giver and taker are energy volumes, joint the joint between them. Two
volumes that share two joints have two rows each way. Ordered by joint, then by
the givers' and takers' order in contacts.
"""
from __future__ import annotations

import numpy as np


def passages(contacts: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Giver, joint and taker of every A -> B -> A passage."""
    order = np.argsort(contacts[:, 1], kind="stable")
    volumes, joints = contacts[order, 0], contacts[order, 1]
    _, start, count = np.unique(joints, return_index=True, return_counts=True)
    row_start = np.repeat(start, count)
    row_count = np.repeat(count, count)
    giver_row = np.repeat(np.arange(len(joints)), row_count)
    within = np.arange(len(giver_row)) - np.repeat(np.cumsum(row_count) - row_count, row_count)
    taker_row = np.repeat(row_start, row_count) + within
    other = giver_row != taker_row
    return volumes[giver_row[other]], joints[giver_row[other]], volumes[taker_row[other]]
