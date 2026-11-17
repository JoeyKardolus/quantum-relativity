"""Tear: an overstretched energy volume tears in two; the place it tears turns from plus to minus and is the new joint.

Contract: `tear(state, held, volume, tear_length)` tears one living energy volume whose
length by the size law is above `tear_length` and that holds at least 3 quanta per box
(648: each of the three cells it becomes needs 1 a box), and returns True; returns False
and does nothing when it is not such a volume. `held` is the host's copy of the per-cell
arrays (events.py), changed here. Axiom 6: only that cell splits.
  - where: along the axis on whose two sides the pull is deepest (the neighbours
    pulling from opposite sides; the first axis when equal)
  - energy: the cell's block has 6 layers along that axis. The quanta of the two
    middle layers, the place it tears, turn from plus to minus and are the new joint's
    (state.torn); the two layers below stay the volume's, the two above are the new
    volume's. Energy has no place, so each third is spread over a whole block, every
    layer over three (whole quanta, exact, phases kept), in the volume's own block and
    in the fresh blocks of two new ids: no box changes cell. When a box holds fewer
    than 3 quanta the layers cannot be shared box by box, and each of the three cells
    gets a third of the whole instead, spread evenly over its block
  - relations: the volume keeps its id and its neighbours on the lower side and round
    the middle, the new volume takes those on the upper side and round the middle, the
    new joint lies between the two and touches the joints round the middle. Every
    neighbour's entries follow, and nothing else about any neighbour changes. On a
    joint's side the volume becomes the half the joint touches, both when the joint
    lies round the middle; the new joint's way is the two halves
The pattern stays A B A B: both halves are wrapped by joints in all 26 directions (the
old joints on their side, the joints round the middle, the new joint between them),
neither touches a volume, and the new joint is part of the web through the joints round
the middle. The new cells start with the volume's pull, clock and heat. Counts
state.tears and state.version, and notes in state.torn_from which volume, along which
axis, the two new ids came from; the caller has made room for two more ids.
"""
from __future__ import annotations

import numpy as np

from physics.engine.state import BOXES, CENTRE, ENERGY_VOLUME, JOINT, State


def tear(state: State, held: dict, volume: int, tear_length: float) -> bool:
    """Tear one too-big energy volume in two with a new joint between the halves."""
    volume = int(volume)
    energy = int(held["cell_quanta"][volume])
    if state.kind_h[volume] != ENERGY_VOLUME or energy < 3 * BOXES:
        return False
    if state.size_energy[ENERGY_VOLUME] / energy <= tear_length:
        return False
    quanta, phase = state.block(volume)
    ids, codes = state.neighbours(volume)
    axis = _axis(held["pull"], ids, codes)
    unit = (9, 3, 1)[axis]
    joint, twin = state.count, state.count + 1
    state.count += 2
    thirds = [_spread(np.moveaxis(quanta, axis, 0)[low:low + 2], np.moveaxis(phase, axis, 0)[low:low + 2], axis)
              for low in (0, 2, 4)]
    if quanta.min() < 3:
        thirds = [(_even(energy // 3 + (energy % 3 > index), quanta.shape), angles)
                  for index, (_, angles) in enumerate(thirds)]
    for cell, sign, (amounts, angles) in zip((volume, joint, twin), (1, -1, 1), thirds):
        state.set_block(cell, sign * amounts, angles)
        held["cell_quanta"][cell] = int(amounts.sum())
        for name in ("pull", "pull_old", "clock", "jiggle"):
            held[name][cell] = held[name][volume]
    state.kind_h[joint], state.kind_h[twin] = JOINT, ENERGY_VOLUME
    side = (codes // unit) % 3 - 1
    round_ids, round_codes = ids[side == 0], codes[side == 0]
    state.set_neighbours(volume, np.concatenate([ids[side <= 0], [joint], round_ids]),
                         np.concatenate([codes[side <= 0], [CENTRE + unit], round_codes + unit]))
    state.set_neighbours(twin, np.concatenate([ids[side >= 0], [joint], round_ids]),
                         np.concatenate([codes[side >= 0], [CENTRE - unit], round_codes - unit]))
    state.set_neighbours(joint, np.concatenate([[volume, twin], round_ids, round_ids, round_ids]),
                         np.concatenate([[CENTRE - unit, CENTRE + unit], round_codes, round_codes + unit,
                                         round_codes - unit]))
    for beside in sorted({int(one) for one in ids}):
        halves = _halves(set(side[ids == beside].tolist()), volume, twin)
        _follow(state, beside, volume, joint, twin, unit)
        first, second = state.sides(beside)
        if volume in first or volume in second:
            state.set_sides(beside, [end for one in first for end in (halves if one == volume else [one])],
                            [end for one in second for end in (halves if one == volume else [one])])
    state.set_sides(joint, [volume], [twin])
    state.torn_from[joint] = state.torn_from[twin] = (volume, axis)
    state.torn += int(thirds[1][0].sum())
    state.tears += 1
    state.version += 1
    return True


def _axis(pull: np.ndarray, ids: np.ndarray, codes: np.ndarray) -> int:
    """The axis on whose two sides the pull is deepest (lowest summed pull of the two face neighbours)."""
    depth = []
    for unit in (9, 3, 1):
        low, high = ids[codes == CENTRE - unit], ids[codes == CENTRE + unit]
        depth.append(float(pull[low].mean() + pull[high].mean()))
    return int(np.argmin(depth))


def _spread(amounts: np.ndarray, angles: np.ndarray, axis: int) -> tuple:
    """Two layers of boxes spread over six: each layer's quanta shared over three layers, exactly; phases repeated."""
    held = np.abs(amounts.astype(np.int64))
    wide = np.repeat(held // 3, 3, axis=0)
    left = np.repeat(held % 3, 3, axis=0)
    order = np.tile(np.arange(3), 2).reshape(6, 1, 1)
    wide += (left > order).astype(np.int64)
    return (np.moveaxis(wide, 0, axis).astype(np.int32),
            np.moveaxis(np.repeat(angles, 3, axis=0), 0, axis).astype(np.float32))


def _even(amount: int, shape: tuple) -> np.ndarray:
    """`amount` quanta shared evenly over a block, the first boxes one more when it does not divide."""
    share = np.full(BOXES, amount // BOXES, dtype=np.int32)
    share[:amount % BOXES] += 1
    return share.reshape(shape)


def _follow(state: State, beside: int, volume: int, joint: int, twin: int, unit: int) -> None:
    """A neighbour's entries for the torn volume follow the halves: lower side, upper side, or all three round the middle."""
    ids, codes = state.neighbours(beside)
    codes = codes.astype(np.int64)
    mine = ids == volume
    side = np.where(mine, ((26 - codes) // unit) % 3 - 1, -1)
    middle = codes[side == 0]
    new_ids = [np.where(side > 0, twin, ids)]
    new_codes = [codes]
    for one, shift in ((twin, 0), (joint, 0), (volume, -unit), (twin, unit), (joint, -unit), (joint, unit)):
        new_ids.append(np.full(len(middle), one, dtype=np.int64))
        new_codes.append(middle + shift)
    state.set_neighbours(beside, np.concatenate(new_ids), np.concatenate(new_codes))


def _halves(sides: set, volume: int, twin: int) -> list:
    """The halves a neighbour touches: the lower (the volume), the upper (the twin), or both when it lies round the middle."""
    if sides <= {-1}:
        return [volume]
    if sides <= {1}:
        return [twin]
    return [volume, twin]
