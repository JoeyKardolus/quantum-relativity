"""fit_measure: what the fit reads off one universe at one moment, from its cells alone.

Contract: fit_measure(universe, settings) -> dict of plain numbers for a physics.engine
State, read from its cells (quanta, lengths, clock rates) and its ways (way_list). The
universe wraps: there is no edge, so everything is measured inside it.
  room          space between lumps, in lengths (bible: the size of space is the sum of
                cell lengths): every joint's length plus the lengths of the least
                packed energy volumes that together hold half the plus quanta (the
                volume the half falls in counted in proportion). The most packed
                volumes holding the other half are the lumps side of the cut; no
                threshold, no cell followed.
  observer_rate the clock rate where the typical quantum sits: the energy volumes' clock
                rates weighted by their quanta
  lump_volumes  energy volumes holding at least LUMP_FACTOR (physics.engine.stats, 2)
                times the start's mean volume quanta
  lumps         groups of lump volumes joined through ways (lumps over regions, not
                single cells); largest_lump the most volumes in one
  lump_share    the lump volumes' share of the plus quanta
  sealed_lumps  lumps whose every way to a volume outside them runs through a sealed
                joint (settings.seal_thin times the start's mean joint quanta or more:
                nothing crosses) and that have such a way; sealed_share their share of
                the plus quanta
  lump_spread   standard deviation of ln(quanta) over the lump volumes (0 with fewer
                than 3): how far the lumps' sizes lie apart
  region_contrast  rms of (quanta in a block / mean - 1) over blocks of 2 x 2 x 2
                lattice energy volumes: unevenness over regions larger than a cell (0
                when the volumes a side are odd or fewer than 4)
  volumes, joints, collapses, tears, crossed, squeeze_max, packing_max
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from physics.engine import ENERGY_VOLUME, JOINT, Settings, State, way_list
from physics.engine.stats import LUMP_FACTOR


def fit_measure(universe: State, settings: Settings) -> dict:
    """Room between lumps, observer clock rate, lumps as groups, seals, spread and region contrast."""
    kind = universe.kind_h[:universe.count]
    held = universe.host("cell_quanta").astype(np.float64)
    length = universe.host("length").astype(np.float64)
    volumes, joints = kind == ENERGY_VOLUME, kind == JOINT
    plus = float(held[volumes].sum())
    packed = volumes & (held >= LUMP_FACTOR * universe.start_quanta[ENERGY_VOLUME])
    carriers, first, second = way_list(universe)
    groups, lumps = _groups(universe.count, packed, first, second)
    lump_quanta = np.bincount(groups[packed], weights=held[packed], minlength=lumps) if lumps else np.zeros(0)
    tight = held[carriers] >= settings.seal_thin * universe.start_quanta[JOINT]
    sealed = _sealed(packed, groups, first, second, tight, lumps)
    side = universe.side
    field = np.where(volumes[:universe.lattice], held[:universe.lattice], 0.0).reshape(side, side, side)[::2, ::2, ::2]
    return {"room": float(length[joints].sum()) + _thin_half(held[volumes], length[volumes]),
            "observer_rate": float((universe.host("rate")[volumes] * held[volumes]).sum() / plus),
            "lump_volumes": int(packed.sum()), "lumps": int(lumps),
            "largest_lump": int(np.bincount(groups[packed]).max()) if lumps else 0,
            "lump_share": float(lump_quanta.sum() / plus),
            "sealed_lumps": int(sealed.sum()), "sealed_share": float(lump_quanta[sealed].sum() / plus),
            "lump_spread": float(np.std(np.log(held[packed]))) if packed.sum() >= 3 else 0.0,
            "region_contrast": _region_contrast(field),
            "volumes": int(volumes.sum()), "joints": int(joints.sum()), "collapses": int(universe.collapses),
            "tears": int(universe.tears), "crossed": int(universe.books.numpy()[1]),
            "squeeze_max": float(universe.host("squeeze")[joints].max()),
            "packing_max": float(universe.host("packing")[volumes].max())}


def _thin_half(quanta: np.ndarray, length: np.ndarray) -> float:
    """Summed length of the least packed volumes (the longest) that together hold half the plus quanta."""
    order = np.argsort(-length, kind="stable")
    before = np.concatenate([[0.0], np.cumsum(quanta[order])[:-1]])
    share = np.clip((0.5 * quanta.sum() - before) / np.maximum(quanta[order], 1.0), 0.0, 1.0)
    return float((share * length[order]).sum())


def _groups(count: int, packed: np.ndarray, first: np.ndarray, second: np.ndarray) -> tuple:
    """Per cell the number of its lump (meaningless for a cell that is not a lump volume), and how many lumps."""
    if not packed.any():
        return np.zeros(count, dtype=np.int64), 0
    joined = packed[first] & packed[second]
    links = coo_matrix((np.ones(int(joined.sum()), dtype=np.int8), (first[joined], second[joined])),
                       shape=(count, count))
    _, labels = connected_components(links, directed=False)
    numbered = np.zeros(count, dtype=np.int64)
    numbered[packed] = np.unique(labels[packed], return_inverse=True)[1]
    return numbered, int(numbered[packed].max()) + 1


def _sealed(packed: np.ndarray, groups: np.ndarray, first: np.ndarray, second: np.ndarray, tight: np.ndarray,
            lumps: int) -> np.ndarray:
    """Per lump: it has a way to the outside and every such way runs through a sealed joint."""
    borders = np.zeros(lumps, dtype=np.int64)
    open_borders = np.zeros(lumps, dtype=np.int64)
    for near, far in ((first, second), (second, first)):
        out = packed[near] & ~(packed[far] & (groups[far] == groups[near]))
        np.add.at(borders, groups[near[out]], 1)
        np.add.at(open_borders, groups[near[out & ~tight]], 1)
    return (borders > 0) & (open_borders == 0)


def _region_contrast(field: np.ndarray) -> float:
    """rms of (block amount / mean - 1) over blocks of 2 x 2 x 2 volumes (0 when the side is odd or under 4)."""
    side = field.shape[0]
    if side % 2 or side < 4:
        return 0.0
    blocks = side // 2
    amount = field.reshape(blocks, 2, blocks, 2, blocks, 2).sum(axis=(1, 3, 5))
    return float(np.sqrt(np.mean((amount / amount.mean() - 1.0) ** 2)))
