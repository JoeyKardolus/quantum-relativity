"""Reconcile joints: make the grid's joints exactly the shared faces of a new tessellation.

Contract: `reconcile_joints(grid, pairs, settings)` returns a new Grid whose
joints are `pairs` ((k, 2), a < b, sorted: Tessellation.pairs), keeping total_space():
  carried   a pair that already was a joint keeps its joint volume.
  vanished  a joint whose face is gone hands its volume back, half to each of its
            two energy volumes' surroundings: their carried joints, by size
            (share_space.py).
  new       a new face starts at its rest volume R (joint_rest.py), half taken
            from each side's surroundings: what vanished joints of that energy volume
            returned, then at most RESERVE (0.4) of its carried joints' volume. When
            that is not enough, the new joints of that energy volume start thinner,
            by the same factor (they are squeezed from the start).
            An energy volume with no carried joint hands what is left of its returned
            space to its new joints, by their rest volume.
  lost      only an energy volume left with no joint at all (an empty cell) loses
            the space its vanished joints returned.
Energy, volume, charge and merged_into are carried unchanged; squeeze is recomputed
(joint_squeeze.py). Pairs must index the grid's energy volumes.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from physics.cells.engine_settings import EngineSettings
from physics.cells.grid import Grid
from physics.cells.joint_rest import joint_rest
from physics.cells.joint_squeeze import joint_squeeze
from physics.cells.share_space import share_space

RESERVE = 0.4


def reconcile_joints(grid: Grid, pairs: np.ndarray, settings: EngineSettings) -> Grid:
    """The grid with joints set to `pairs`, joint volumes carried, returned and taken."""
    count = grid.cell_count
    old_keys = _keys(np.sort(grid.joints, axis=1), count)
    new_keys = _keys(pairs, count)
    carried = np.isin(new_keys, old_keys)
    order = np.argsort(old_keys)
    found = order[np.searchsorted(old_keys, new_keys[carried], sorter=order)]
    gone = ~np.isin(old_keys, new_keys)
    returned = _to_both(grid.joints[gone], grid.joint_volume[gone] / 2, count)
    kept_pairs = pairs[carried]
    kept_volume = grid.joint_volume[found]
    around = _to_both(kept_pairs, kept_volume, count)
    fresh_pairs = pairs[~carried]
    half_rest = joint_rest(grid.volume, fresh_pairs, settings.joint_rest) / 2
    demand = _to_both(fresh_pairs, half_rest, count)
    supply = returned + RESERVE * around
    scale = np.minimum(1.0, np.divide(supply, demand, out=np.ones(count), where=demand > 0))
    taken = _to_both(fresh_pairs, half_rest, count, scale)
    fresh_volume = half_rest * (scale[fresh_pairs[:, 0]] + scale[fresh_pairs[:, 1]])
    leftover = np.where(around > 0, 0.0, returned - taken)
    fresh_volume += _hand_out(fresh_pairs, 2 * half_rest, leftover, count)
    joint_volume = np.empty(len(pairs))
    joint_volume[carried] = share_space(kept_pairs, kept_volume,
                                        np.where(around > 0, returned - taken, 0.0))
    joint_volume[~carried] = fresh_volume
    reconciled = replace(grid, joints=pairs, joint_volume=joint_volume)
    return replace(reconciled, squeeze=joint_squeeze(reconciled, settings))


def _keys(pairs: np.ndarray, count: int) -> np.ndarray:
    return pairs[:, 0].astype(np.int64) * count + pairs[:, 1]


def _to_both(pairs: np.ndarray, amount: np.ndarray, count: int,
             scale: np.ndarray | None = None) -> np.ndarray:
    """Per energy volume, the sum of `amount` over its pairs (times its own `scale`)."""
    if scale is None:
        scale = np.ones(count)
    return (np.bincount(pairs[:, 0], amount * scale[pairs[:, 0]], count)
            + np.bincount(pairs[:, 1], amount * scale[pairs[:, 1]], count))


def _hand_out(pairs: np.ndarray, rest: np.ndarray, space: np.ndarray, count: int) -> np.ndarray:
    """Each energy volume's `space` spread over its pairs in proportion to `rest`."""
    total = _to_both(pairs, rest, count)
    rate = np.divide(space, total, out=np.zeros(count), where=total > 0)
    return rest * (rate[pairs[:, 0]] + rate[pairs[:, 1]])
