"""fit_measure: what the fit reads off one universe at one moment, from its cells alone.

Contract: fit_measure(universe, settings) -> dict of plain numbers for a physics.engine
State, read from its cells (quanta, lengths, clock rates) and its ways (way_list). The
universe wraps: there is no edge, so everything is measured inside it.
  size          the size of the universe: the sum of every living cell's length
  observer_rate the clock rate where the typical quantum sits: the energy volumes' clock
                rates weighted by their quanta
  alive         the share of the start's energy volumes that are still an energy volume
                (one that fell into another at a joint collapse is gone): 1 at the
                start, 1 / volumes when everything fell into a single cell
  spent_share   the share of the size lying in energy volumes that are overstretched:
                longer than settings.tear_length (about to tear, or too empty to), or
                holding fewer than SPENT_BOXES (3) quanta per Planck box, the engine's
                own floor, under which a volume can neither tear nor be read as a
                fluid. 0 in a calm universe; a tear chain runs through overstretched
                volumes down to the floor and takes it toward 1
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
  plus, minus, amount   the quanta in energy volumes, in joints, and both together
  volumes, joints, collapses, merges, tears, crossed, squeeze_max, packing_max
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from physics.engine import BOXES, ENERGY_VOLUME, JOINT, Settings, State, way_list
from physics.engine.stats import LUMP_FACTOR

SPENT_BOXES = 3


def fit_measure(universe: State, settings: Settings) -> dict:
    """Size, observer clock rate, what is left alive and spent, lumps as groups, seals, spread and region contrast."""
    kind = universe.kind_h[:universe.count]
    held = universe.host("cell_quanta").astype(np.float64)
    length = universe.host("length").astype(np.float64)
    volumes, joints = kind == ENERGY_VOLUME, kind == JOINT
    plus, size = float(held[volumes].sum()), float(length[kind != 0].sum())
    packed = volumes & (held >= LUMP_FACTOR * universe.start_quanta[ENERGY_VOLUME])
    carriers, first, second = way_list(universe)
    groups, lumps = _groups(universe.count, packed, first, second)
    lump_quanta = np.bincount(groups[packed], weights=held[packed], minlength=lumps) if lumps else np.zeros(0)
    tight = held[carriers] >= settings.seal_thin * universe.start_quanta[JOINT]
    sealed = _sealed(packed, groups, first, second, tight, lumps)
    side = universe.side
    places = volumes[:universe.lattice]
    field = np.where(places, held[:universe.lattice], 0.0).reshape(side, side, side)[::2, ::2, ::2]
    return {"size": size, "observer_rate": float((universe.host("rate")[volumes] * held[volumes]).sum() / plus),
            "alive": float(places.sum() / (side // 2) ** 3),
            "spent_share": float(length[volumes & ((held < SPENT_BOXES * BOXES) | _torn(length, settings))].sum() / size),
            "lump_volumes": int(packed.sum()), "lumps": int(lumps),
            "largest_lump": int(np.bincount(groups[packed]).max()) if lumps else 0,
            "lump_share": float(lump_quanta.sum() / plus),
            "sealed_lumps": int(sealed.sum()), "sealed_share": float(lump_quanta[sealed].sum() / plus),
            "lump_spread": float(np.std(np.log(held[packed]))) if packed.sum() >= 3 else 0.0,
            "region_contrast": _region_contrast(field),
            "plus": plus, "minus": float(held[joints].sum()), "amount": float(held[kind != 0].sum()),
            "volumes": int(volumes.sum()), "joints": int(joints.sum()), "collapses": int(universe.collapses),
            "merges": int(universe.merges), "tears": int(universe.tears), "crossed": int(universe.books.numpy()[1]),
            "squeeze_max": float(universe.host("squeeze")[joints].max()),
            "packing_max": float(universe.host("packing")[volumes].max())}


def _torn(length: np.ndarray, settings: Settings) -> np.ndarray:
    """Per cell: longer than the tear length (never, when tears are off)."""
    return (length > settings.tear_length) if settings.tear_length > 0.0 else np.zeros(length.shape, dtype=bool)


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
