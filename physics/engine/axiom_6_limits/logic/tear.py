"""Tear: an overstretched energy volume tears in two, and the tear's place is the new joint.

The quanta of the boxes that touch the tear's plane flip from plus to minus and are the
joint, the least it will ever hold: at the least boxes across the two layers either side,
the middle third of the cell at any boxes. The fluid on either side is a half, each
laid as a ball whirl again, the far half turned half round about the line between them. A
joint on the far side goes with the far half; a joint along the side is cut (`cut.py`).
A tear takes a turn of the whirl: a volume tears a calm swirl's turn after it was made, or
later. Found on the host, from the cells' totals, before the fluid moves.
"""
import numpy as np

from physics.engine.axiom_3_structure.definitions.kinds import QUANTA_A_BOX
from physics.engine.axiom_3_structure.definitions.relations import join, joints_of
from physics.engine.axiom_3_structure.definitions.store import Store, slot_of
from physics.engine.axiom_3_structure.logic.take_own import take_own
from physics.engine.axiom_6_limits.definitions.overstretched import (
    TEAR_SIZE, half_round, layers, least_whirl, torn_across)
from physics.engine.axiom_6_limits.logic.cut import new_cell, pass_on, relay


def _cube(store: Store, slot: int) -> np.ndarray:
    """The quanta of the own cell in `slot` as a block."""
    return store.box_quanta[slot].numpy().reshape((store.across,) * 3).astype(np.int64)


def _pulls(ends: np.ndarray, volume: int, pull_of: dict) -> np.ndarray:
    """The pull on the volume's near and far side along each direction, from its joints'."""
    pulls = np.zeros((3, 2))
    for joint, end in joints_of(ends, volume):
        pulls[ends[joint, 2], 1 - end] += pull_of.get(joint, 0.0)  # named first: on its far side
    return pulls


def tear_volume(store: Store, volume: int, way: int, cube: np.ndarray = None) -> bool:
    """Tear one volume across direction `way`; False when a half would be no whirl."""
    cube = _cube(store, slot_of(store, volume)) if cube is None else cube
    held, third = layers(cube, way), store.across // 3
    near, flipped, far = held[:third].sum(), held[third:-third].sum(), held[-third:].sum()
    if min(near, far) < least_whirl(store.across):
        return False
    laid = int(store.cell_laid.numpy()[volume])
    other = new_cell(store, int(far), half_round(laid, way))
    joint = new_cell(store, -int(flipped), laid)
    relay(store, volume, int(near))
    pass_on(store, volume, other, way)
    join(store, joint, volume, other, way)
    store.tears += 1
    return True


def tear(store: Store) -> None:
    """Tear every volume that is overstretched and a whirl's turn old, each where it is stretched."""
    held = store.cell_quanta.numpy()[:store.cells]
    age = store.step - store.cell_born.numpy()[:store.cells]
    big = (held >= 2 * least_whirl(store.across)) & (held * TEAR_SIZE < QUANTA_A_BOX * store.across ** 3)
    ripe = np.flatnonzero(big & (age * store.turn_per_step >= 2 * np.pi)).tolist()
    for volume in ripe:
        take_own(store, volume)  # a calm volume tears from its rule's boxes
    if not ripe:
        return
    own, ends = store.own_cell.numpy()[:store.count].tolist(), store.joint_ends.numpy()[:store.cells]
    pull_of = dict(zip(own, store.own_squeeze.numpy()[:store.count, 0].tolist()))
    cubes = [_cube(store, own.index(volume)) for volume in ripe]  # all read before any tears
    ways = [torn_across(cube, _pulls(ends, volume, pull_of)) for volume, cube in zip(ripe, cubes)]
    for volume, way, cube in zip(ripe, ways, cubes):
        tear_volume(store, volume, way, cube)
