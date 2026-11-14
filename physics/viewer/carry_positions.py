"""Carry display positions across a step in which joints may have collapsed.

Contract: `carry_positions(positions, weight, merged_into, joint_count)`
returns (joint_count, 3) positions for the stepped network: new joint k sits
at the weight-weighted mean of the old joints with merged_into == k (pass each
old joint's energy, so a merged joint sits at the energy-weighted mean of its
parts). merged_into None means nothing collapsed: positions come back as they
were. Display only.
"""
from __future__ import annotations

import numpy as np


def carry_positions(positions: np.ndarray, weight: np.ndarray, merged_into: np.ndarray | None,
                    joint_count: int) -> np.ndarray:
    """Old display positions mapped onto the new joints, merged joints energy-weighted."""
    if merged_into is None:
        return positions
    weight = np.maximum(weight, 1e-12)
    total = np.bincount(merged_into, weight, joint_count)
    return np.stack([np.bincount(merged_into, weight * positions[:, axis], joint_count)
                     for axis in range(3)], axis=1) / np.maximum(total, 1e-300)[:, None]
