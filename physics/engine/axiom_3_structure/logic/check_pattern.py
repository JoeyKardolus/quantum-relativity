"""Check pattern: whether the A B A B pattern holds, counted from the relation tables.

Contract: `check_pattern(state)` reads the tables the kernels read (the neighbour pools
and the way table as they stand on the device, `kind`, `made_kind`) and returns a dict
of whole numbers. It writes nothing. All of these are 0 while the pattern holds:
  touching       energy volumes with a neighbour that is not a living joint (axiom 3:
                 plus never touches plus)
  bare           living energy volumes with no living joint (minus wraps every plus)
  one_sided      neighbour entries `(cell -> other, code)` between living cells whose
                 mirror `(other -> cell, opposite code)` is missing
  dead_links     neighbour entries of living cells that name an id that is no cell
  loose_ways     living ways not made by two opposite entries of their joint (or held
                 in two slots)
  missing_ways   pairs of opposite entries of a joint, two different energy volumes,
                 with no way in the table
  shared_mouths  ways beyond the first that use one mouth of one volume in one pass
  flickered      living ids whose kind is not the kind they were made with (the
                 pattern never flickers)
  lone_joints    living joints that touch no other living joint
and one that is 1 while the pattern holds:
  web_pieces     groups of living joints, joined only where two joints are neighbours
                 of each other: 1 while minus forms one web (0 when there is no joint)
Two more are measured, not judged (a cell has no geometry, so nothing says how many
joints a wrap must have or that a joint must lie between two different volumes):
  fewest_joints  the fewest different living joints round any living energy volume (26
                 at the start; 0 when there is no volume)
  inner_joints   living joints with no way: every volume such a joint touches is one
                 and the same, because the volumes it kept apart have merged round it
  crowded_mouths mouths of a volume that more than one way goes through (after a tear
                 two pieces lie on one side of the joints beside them); such a mouth
                 sends once a step, shared between its ways (`cross`, axiom 4)
A side is a relation, not a place: the check never counts direction codes, so listing
one joint under several codes makes no count better.
Only cells that live have their entries read; an id that is no cell has no relations.
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from physics.engine.axiom_3_structure.definitions.direction_codes import CENTRE
from physics.engine.axiom_3_structure.definitions.kinds import ENERGY_VOLUME, JOINT, NO_CELL


def _living_entries(state, kind: np.ndarray) -> tuple:
    """Every neighbour entry of every living cell as three flat arrays: owner, other, code."""
    number = np.where(kind != NO_CELL, state.host("adj_n"), 0).astype(np.int64)
    owner = np.repeat(np.arange(kind.size, dtype=np.int64), number)
    within = np.arange(owner.size, dtype=np.int64) - np.repeat(np.cumsum(number) - number, number)
    where = state.host("adj_at").astype(np.int64)[owner] + within
    return owner, state.adj_id.numpy()[where].astype(np.int64), state.adj_code.numpy()[where].astype(np.int64)


def _names_living(other: np.ndarray, kind: np.ndarray) -> np.ndarray:
    """Per entry: whether the id it names is a living cell."""
    known = (other >= 0) & (other < kind.size)
    return known & (kind[np.where(known, other, 0)] != NO_CELL)


def _unmirrored(owner: np.ndarray, other: np.ndarray, code: np.ndarray, count: int) -> int:
    """How many entries have no mirror entry `(other -> owner, opposite code)`."""
    keys = (owner * count + other) * 27 + code
    mirrors = (other * count + owner) * 27 + (26 - code)
    return int(np.count_nonzero(~np.isin(mirrors, keys)))


def _wanted_ways(joint: np.ndarray, volume: np.ndarray, code: np.ndarray) -> set:
    """The ways the joints' own entries make, `(a, b, mouth_a, mouth_b, joint)` each.

    `joint`, `volume`, `code` are the entries of living joints that name a living energy
    volume. Every entry at a code below the centre is paired with every entry of the
    same joint at the opposite code; two entries of one volume make no way.
    """
    low = code < CENTRE
    high = np.flatnonzero(code > CENTRE)
    high = high[np.argsort(joint[high] * 27 + code[high], kind="stable")]
    high_keys = joint[high] * 27 + code[high]
    facing = joint[low] * 27 + (26 - code[low])
    first = np.searchsorted(high_keys, facing, side="left")
    pairs = np.searchsorted(high_keys, facing, side="right") - first
    low_row = np.repeat(np.flatnonzero(low), pairs)
    high_row = high[np.repeat(first, pairs) + np.arange(low_row.size) - np.repeat(np.cumsum(pairs) - pairs, pairs)]
    one, two, at = volume[low_row], volume[high_row], code[low_row]
    lower = one < two
    rows = np.stack([np.where(lower, one, two), np.where(lower, two, one), np.where(lower, 26 - at, at),
                     np.where(lower, at, 26 - at), joint[low_row]], axis=1)[one != two]
    return set(map(tuple, rows.tolist()))


def _living_ways(state) -> np.ndarray:
    """The living rows of the way table: `(a, b, mouth_a, mouth_b, joint, pass)` each."""
    table = np.stack([state.host(name) for name in ("way_a", "way_b", "way_mouth_a", "way_mouth_b", "way_joint",
                                                    "way_pass")], axis=1).astype(np.int64)
    return table[table[:, 4] >= 0]


def _shared_mouths(ways: np.ndarray) -> int:
    """How many ways beyond the first use one mouth of one volume in one pass."""
    if not len(ways):
        return 0
    passes = int(ways[:, 5].max()) + 2
    mouths = np.concatenate([(ways[:, 0] * 27 + ways[:, 2]) * passes + ways[:, 5] + 1,
                             (ways[:, 1] * 27 + ways[:, 3]) * passes + ways[:, 5] + 1])
    return int(mouths.size - np.unique(mouths).size)


def _crowded_mouths(ways: np.ndarray) -> int:
    """How many mouths of a volume have more than one way going through them."""
    if not len(ways):
        return 0
    mouths = np.concatenate([ways[:, 0] * 27 + ways[:, 2], ways[:, 1] * 27 + ways[:, 3]])
    return int(np.count_nonzero(np.unique(mouths, return_counts=True)[1] > 1))


def _web_pieces(owner: np.ndarray, other: np.ndarray, kind: np.ndarray) -> int:
    """How many groups the living joints fall into, joined only by entries from joint to joint."""
    joined = coo_matrix((np.ones(owner.size, dtype=np.int8), (owner, other)), shape=(kind.size, kind.size))
    labels = connected_components(joined, directed=False)[1]
    return int(np.unique(labels[kind == JOINT]).size)


def _fewest_joints(owner: np.ndarray, other: np.ndarray, kind: np.ndarray) -> int:
    """The fewest different living joints round any living energy volume; 0 when there is no volume."""
    volumes = kind == ENERGY_VOLUME
    if not volumes.any():
        return 0
    pairs = np.unique(owner * kind.size + other)
    round_each = np.bincount(pairs // kind.size, minlength=kind.size)[:kind.size]
    return int(round_each[volumes].min())


def check_pattern(state) -> dict:
    """Count every way the pattern can be broken; all 0 (and one web piece) while it holds."""
    kind = state.host("kind")
    count = kind.size
    owner, other, code = _living_entries(state, kind)
    alive = _names_living(other, kind)
    volume_owner = kind[owner] == ENERGY_VOLUME
    to_joint = alive & (kind[np.where(alive, other, 0)] == JOINT)
    to_volume = alive & (kind[np.where(alive, other, 0)] == ENERGY_VOLUME)
    from_joint = (kind[owner] == JOINT) & to_volume
    ways = _living_ways(state)
    have = list(map(tuple, ways[:, :5].tolist()))
    wanted = _wanted_ways(owner[from_joint], other[from_joint], code[from_joint])
    wrapped = np.unique(owner[volume_owner & to_joint]).size
    with_way = np.bincount(ways[:, 4], minlength=count)[:count] > 0
    joint_to_joint = (kind[owner] == JOINT) & to_joint
    meets_joint = np.bincount(owner[joint_to_joint], minlength=count)[:count] > 0
    return {
        "touching": int(np.unique(owner[volume_owner & ~to_joint]).size),
        "bare": int(np.count_nonzero(kind == ENERGY_VOLUME) - wrapped),
        "one_sided": _unmirrored(owner[alive], other[alive], code[alive], count),
        "dead_links": int(np.count_nonzero(~alive)),
        "loose_ways": len(have) - len(wanted & set(have)),
        "missing_ways": len(wanted - set(have)),
        "shared_mouths": _shared_mouths(ways),
        "flickered": int(np.count_nonzero((kind != NO_CELL) & (kind != state.host("made_kind")))),
        "lone_joints": int(np.count_nonzero((kind == JOINT) & ~meets_joint)),
        "web_pieces": _web_pieces(owner[joint_to_joint], other[joint_to_joint], kind),
        "fewest_joints": _fewest_joints(owner[volume_owner & to_joint], other[volume_owner & to_joint], kind),
        "crowded_mouths": _crowded_mouths(ways),
        "inner_joints": int(np.count_nonzero((kind == JOINT) & ~with_way)),
    }
