"""fit_measure: what the fit reads off one universe at one moment, measured inside it, never at an edge.

Contract: fit_measure(universe, settings, block) -> dict of plain numbers for universe 0 of a
wrapping State (physics.engine), read from its cells (physics.engine.cells) and tallies:
  room          space between lumps, in Planck boxes (bible rule 1: size is counted in
                boxes): every joint's boxes plus the boxes of the least packed energy
                volumes that together hold half the plus quanta (the volume the half
                falls in counted in proportion). The most packed volumes holding the
                other half are the lumps side of the cut; no threshold, no cell followed.
  observer_rate the clock rate where the typical quantum sits: the energy volumes' clock
                rates (1 / (1 + packing / clock_packing)), weighted by their quanta
  lump_volumes  energy volumes packed at least LUMP_FACTOR (physics.engine.stats, 2)
                times the start's mean plus quanta per plus box
  lumps         groups of lump volumes joined through shared joints (lumps over
                regions, not single cells); largest_lump the most volumes in one
  lump_share    the lump volumes' share of the plus quanta
  sealed_lumps  lumps whose every joint to a volume outside them is squeezed to the
                seal or past it (settings.seal_squeeze: nothing crosses) and that have
                such a joint; sealed_share their share of the plus quanta
  lump_spread   standard deviation of ln(quanta) over the lump volumes (0 with fewer
                than 3): how far the lumps' sizes lie apart
  region_contrast  rms of (quanta in a block / mean - 1) over blocks `block` boxes
                across, both signs (the fit passes two start cells): unevenness over
                regions larger than a cell; 0 when the side is not a multiple of block
  volumes, joints, collapses, splits, crossed, squeeze_max, packing_max, capped
Raises ValueError for a universe that does not wrap (it has an edge).
"""
from __future__ import annotations

import numpy as np

from physics.engine import Settings, State, cells, host_boxes
from physics.engine.stats import LUMP_FACTOR


def fit_measure(universe: State, settings: Settings, block: int) -> dict:
    """Room between lumps, observer clock rate, lumps as groups, seals, spread and region contrast of universe 0."""
    if not universe.wrap:
        raise ValueError("the fit measures space inside a wrapping universe; this one has an edge")
    boxes = host_boxes(universe)
    found = cells(boxes)
    count = found["volumes"]
    held = np.maximum(found["boxes"], 1).astype(np.float64)
    quanta = found["quanta"].astype(np.float64)
    packing = quanta[:count] / held[:count]
    plus = float(quanta[:count].sum())
    start = universe.start[0]
    rate = 1.0 / (1.0 + packing / settings.clock_packing) if settings.clock_packing > 0.0 else np.ones(count)
    packed = packing >= LUMP_FACTOR * start["plus"] / start["plus_boxes"]
    groups = _groups(found, packed)
    squeeze = _squeeze(found, held, float(boxes["size_energy"][1]))
    lump_quanta = np.bincount(groups[packed], weights=quanta[:count][packed]) if packed.any() else np.zeros(0)
    sealed = _sealed(found, packed, groups, squeeze, settings.seal_squeeze, lump_quanta.size)
    return {"room": float(found["boxes"][count:].sum()) + _thin_half_boxes(packing, quanta[:count], held[:count]),
            "observer_rate": float((rate * quanta[:count]).sum() / plus),
            "lump_volumes": int(packed.sum()), "lumps": int(lump_quanta.size),
            "largest_lump": int(np.bincount(groups[packed]).max()) if packed.any() else 0,
            "lump_share": float(lump_quanta.sum() / plus),
            "sealed_lumps": int(sealed.sum()), "sealed_share": float(lump_quanta[sealed].sum() / plus),
            "lump_spread": float(np.std(np.log(quanta[:count][packed]))) if packed.sum() >= 3 else 0.0,
            "region_contrast": _region_contrast(boxes["quanta"], block),
            "volumes": int(count), "joints": int(found["kind"].size - count),
            "collapses": int(universe.collapses[0]), "splits": int(universe.splits[0]),
            "crossed": int(universe.tallies.numpy()[0][0]), "squeeze_max": float(squeeze.max()) if squeeze.size else 0.0,
            "packing_max": float(packing.max()), "capped": int(universe.capped[0])}


def _thin_half_boxes(packing: np.ndarray, quanta: np.ndarray, held: np.ndarray) -> float:
    """Boxes of the least packed volumes that together hold half the plus quanta."""
    order = np.argsort(packing, kind="stable")
    before = np.concatenate([[0.0], np.cumsum(quanta[order])[:-1]])
    share = np.clip((0.5 * quanta.sum() - before) / np.maximum(quanta[order], 1.0), 0.0, 1.0)
    return float((share * held[order]).sum())


def _groups(found: dict, packed: np.ndarray) -> np.ndarray:
    """Per volume the number of its lump (0 .. lumps - 1; meaningless for a volume that is not packed)."""
    count = found["volumes"]
    root = np.arange(count)
    ends = found["between"][count:]
    joined = ends[(ends >= 0).all(axis=1)]
    joined = joined[packed[joined[:, 0]] & packed[joined[:, 1]]]
    while True:
        lower = np.minimum(root[joined[:, 0]], root[joined[:, 1]])
        fresh = root.copy()
        np.minimum.at(fresh, joined[:, 0], lower)
        np.minimum.at(fresh, joined[:, 1], lower)
        fresh = fresh[fresh]
        if np.array_equal(fresh, root):
            break
        root = fresh
    numbered = np.zeros(count, dtype=np.int64)
    numbered[packed] = np.unique(root[packed], return_inverse=True)[1]
    return numbered


def _squeeze(found: dict, held: np.ndarray, constant: float) -> np.ndarray:
    """Each joint's squeeze, the length its energy wants over its length, minus 1 (as physics.engine.stats)."""
    count = found["volumes"]
    wanted = (constant / np.maximum(found["quanta"][count:], 1).astype(np.float64)) ** 3
    return np.cbrt(wanted / np.maximum(found["size"][count:], 1e-6)) - 1.0


def _sealed(found: dict, packed: np.ndarray, groups: np.ndarray, squeeze: np.ndarray, seal: float,
            lumps: int) -> np.ndarray:
    """Per lump: it has a joint to the outside and every such joint is squeezed to the seal or past it."""
    count = found["volumes"]
    ends = found["between"][count:]
    whole = (ends >= 0).all(axis=1)
    borders = np.zeros(lumps, dtype=np.int64)
    open_borders = np.zeros(lumps, dtype=np.int64)
    for near, far in ((0, 1), (1, 0)):
        inside = whole & packed[ends[:, near]] & ~(packed[ends[:, far]] & (groups[ends[:, far]] == groups[ends[:, near]]))
        np.add.at(borders, groups[ends[inside, near]], 1)
        np.add.at(open_borders, groups[ends[inside & (squeeze < seal), near]], 1)
    return (borders > 0) & (open_borders == 0)


def _region_contrast(quanta: np.ndarray, period: int) -> float:
    """rms of (block amount / mean - 1) over blocks `period` boxes across (0 when the side does not divide)."""
    side = quanta.shape[0]
    if period < 1 or side % period or side // period < 2:
        return 0.0
    blocks = side // period
    amount = np.abs(quanta.astype(np.int64)).reshape(blocks, period, blocks, period, blocks, period).sum(axis=(1, 3, 5))
    return float(np.sqrt(np.mean((amount / amount.mean() - 1.0) ** 2)))
