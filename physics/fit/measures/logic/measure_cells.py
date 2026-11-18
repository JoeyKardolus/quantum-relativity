"""Measure cells: what the fit reads off one moment of a universe, from its cells alone.

Contract: `measure_cells(cells, tear_length)` takes a `read_universe` dict and the
engine's `tear_length` setting and returns plain numbers. The universe wraps: there is
no edge, so everything is measured inside it.
  size           the size of the universe: the sum of the lengths of every living cell
                 that holds energy (a cell holding nothing is endlessly long and is left
                 out, as `physics.engine.measure_grid` leaves it out, and counted)
  endless        living cells holding 0 quanta
  observer_rate  the clock rate where the typical quantum sits: the energy volumes'
                 clock rates weighted by their quanta
  alive          the share of the start's energy volumes that are still an energy volume
                 (one that fell into another at a joint collapse is gone): 1 at the
                 start, 1 / volumes when everything fell into a single cell
  volume_size, joint_size   the size's two parts
  spent_share    the share of the energy volumes' size lying in overstretched volumes
                 (longer than `tear_length`): 0 in a calm universe; a tear chain takes
                 it toward 1. With an endless energy volume that size is endless and all
                 of it lies in that volume, so it is exactly 1. It is a share of the
                 volumes' size, not of the whole: by the one size law a default joint
                 starts 30 times longer than a volume, so the joints hold 99% of the
                 whole size and would hide a tear chain
  lump_volumes   energy volumes holding at least LUMP_FACTOR times the start's mean
  lumps          groups of lump volumes joined through ways; largest_lump the most
                 volumes in one
  lump_share     the lump volumes' share of the plus quanta
  lump_spread    standard deviation of ln(quanta) over the lump volumes (0 with fewer
                 than SPREAD_LUMPS): how far the lumps' sizes lie apart
  region_contrast   `region_contrast` of the start lattice's volumes
  plus, minus, amount   the quanta in energy volumes, in joints, and both together
  volumes, joints, collapses, merges, tears, squeeze_max, packing_max
Raises ValueError when the energy volumes hold no quanta: there is nothing to measure.
"""
from __future__ import annotations

import numpy as np

from physics.engine import ENERGY_VOLUME, JOINT, NO_CELL
from physics.fit.measures.definitions.lump_groups import lump_groups
from physics.fit.measures.definitions.lump_marks import SPREAD_LUMPS
from physics.fit.measures.definitions.lump_volumes import lump_volumes
from physics.fit.measures.definitions.overstretched import overstretched
from physics.fit.measures.definitions.region_contrast import region_contrast


def _spent_share(volumes: np.ndarray, held: np.ndarray, length: np.ndarray, tear_length: float) -> float:
    """The share of the volumes' size in overstretched energy volumes; 1 when one of them is endless."""
    if (volumes & (held <= 0)).any():
        return 1.0
    return float(length[volumes & overstretched(length, tear_length)].sum() / length[volumes].sum())


def _lump_numbers(cells: dict, packed: np.ndarray) -> dict:
    """How many lump volumes and lumps there are, the largest lump, their share and their spread."""
    held, plus = cells["held"], int(cells["held"][cells["kind"] == ENERGY_VOLUME].sum())
    groups, lumps = lump_groups(cells["kind"].size, packed, cells["ways"][1], cells["ways"][2])
    return {"lump_volumes": int(packed.sum()), "lumps": int(lumps),
            "largest_lump": int(np.bincount(groups[packed]).max()) if lumps else 0,
            "lump_share": float(held[packed].sum() / plus),
            "lump_spread": float(np.std(np.log(held[packed]))) if packed.sum() >= SPREAD_LUMPS else 0.0}


def measure_cells(cells: dict, tear_length: float) -> dict:
    """Size, observer rate, what is alive and spent, the lumps and the region contrast of one moment."""
    kind, held, length, side = cells["kind"], cells["held"], cells["length"], cells["side"]
    volumes, joints = kind == ENERGY_VOLUME, kind == JOINT
    plus = int(held[volumes].sum())
    if plus <= 0:
        raise ValueError("the energy volumes hold no quanta: nothing to measure")
    volume_size = float(length[volumes & (held > 0)].sum())
    joint_size = float(length[joints & (held > 0)].sum())
    places = volumes[:side ** 3]
    field = np.where(places, held[:side ** 3], 0).reshape(side, side, side)[::2, ::2, ::2]
    return {"size": volume_size + joint_size, "volume_size": volume_size, "joint_size": joint_size,
            "endless": int(((kind != NO_CELL) & (held <= 0)).sum()),
            "observer_rate": float((cells["rate"][volumes] * held[volumes]).sum() / plus),
            "alive": float(places.sum() / (side // 2) ** 3),
            "spent_share": _spent_share(volumes, held, length, tear_length),
            **_lump_numbers(cells, lump_volumes(kind, held, cells["start_volume_quanta"])),
            "region_contrast": region_contrast(field),
            "plus": plus, "minus": int(held[joints].sum()), "amount": int(held[kind != NO_CELL].sum()),
            "volumes": int(volumes.sum()), "joints": int(joints.sum()), "collapses": cells["collapses"],
            "merges": cells["merges"], "tears": cells["tears"], "squeeze_max": cells["squeeze_max"],
            "packing_max": cells["packing_max"]}
