"""Measure grid: how big the grid is and how far its lumps warp it.

Contract: `measure_grid(state)` reads `length`, `kind`, `cell_quanta` and the way table and returns
plain Python numbers. It writes nothing.
  size             the sum of the lengths of every living cell that holds energy
  volume_size, joint_size   its two parts
  cells, volumes, joints    the living cells, and by kind
  endless          living cells holding 0 quanta: endlessly long by the size law, so
                   left out of every sum, mean, shortest and longest here, and counted
  shortest_volume, longest_volume, shortest_joint, longest_joint   lengths (0.0 when
                   there is no such cell holding energy)
  warp             how much shorter the way from a lump to its neighbours is than the
                   usual way: the mean reach (half the one volume's length, the joint's
                   length, half the other volume's) of the ways of the volumes holding
                   the top tenth of the plus energy (the fewest volumes, fullest first, a
                   tie to the lower id), over the mean reach of all ways. Below 1: the
                   grid is pulled together round its lumps. 1 in an even grid. `nan`
                   when there is no way to measure along
  lump_share       the share of the volumes that hold the top tenth of the plus energy (0.1
                   when every volume holds the same; near 0 when a few hold it)
"""
from __future__ import annotations

import numpy as np

from physics.engine.axiom_3_structure.definitions.kinds import ENERGY_VOLUME, JOINT, NO_CELL


def _span(lengths: np.ndarray) -> tuple:
    """The shortest and the longest of some lengths; 0.0 twice for none."""
    if not lengths.size:
        return 0.0, 0.0
    return float(lengths.min()), float(lengths.max())


def _fullest(quanta: np.ndarray) -> np.ndarray:
    """Whether each volume is among the fewest, fullest first, that hold a tenth of the plus energy."""
    top = np.zeros(quanta.size, dtype=bool)
    if not quanta.size or quanta.sum() <= 0:
        return top
    fullest_first = np.argsort(-quanta, kind="stable")
    held = np.cumsum(quanta[fullest_first])
    top[fullest_first[:int(np.searchsorted(held * 10, held[-1], side="left")) + 1]] = True
    return top


def _warp(state, length: np.ndarray, fullest: np.ndarray) -> float:
    """Mean reach of the fullest volumes' ways over the mean reach of all ways."""
    joint = state.host("way_joint")
    living = joint >= 0
    way_a, way_b, joint = state.host("way_a")[living], state.host("way_b")[living], joint[living]
    reach = length[way_a] / 2.0 + length[joint] + length[way_b] / 2.0
    finite = np.isfinite(reach)
    lump = finite & (fullest[way_a] | fullest[way_b])
    if not lump.any():
        return float("nan")
    return float(reach[lump].mean() / reach[finite].mean())


def measure_grid(state) -> dict:
    """The grid's size, its cells by kind, their shortest and longest, and the warp."""
    kind = state.host("kind")
    quanta = state.host("cell_quanta").astype(np.int64)
    length = state.host("length").astype(np.float64)
    volumes = (kind == ENERGY_VOLUME) & (quanta > 0)
    fullest = np.zeros(kind.size, dtype=bool)
    fullest[np.flatnonzero(volumes)] = _fullest(quanta[volumes])
    joints = (kind == JOINT) & (quanta > 0)
    shortest_volume, longest_volume = _span(length[volumes])
    shortest_joint, longest_joint = _span(length[joints])
    volume_size, joint_size = float(length[volumes].sum()), float(length[joints].sum())
    return {
        "size": volume_size + joint_size,
        "volume_size": volume_size,
        "joint_size": joint_size,
        "cells": int(np.count_nonzero(kind != NO_CELL)),
        "volumes": int(np.count_nonzero(kind == ENERGY_VOLUME)),
        "joints": int(np.count_nonzero(kind == JOINT)),
        "endless": int(np.count_nonzero((kind != NO_CELL) & (quanta <= 0))),
        "shortest_volume": shortest_volume,
        "longest_volume": longest_volume,
        "shortest_joint": shortest_joint,
        "longest_joint": longest_joint,
        "warp": _warp(state, length, fullest),
        "lump_share": float(fullest.sum() / volumes.sum()) if volumes.any() else float("nan"),
    }
