"""Joints of: the live joints of a store and the two volumes each lies between, read on the host.

A joint's place gives its direction and its two volumes by arithmetic (`places.py`); a
joint that has collapsed holds plus and is no joint any more.
"""
import numpy as np

from physics.engine.axiom_3_structure.definitions.places import steps_of
from physics.engine.axiom_3_structure.definitions.store import TILE, Store


def slots_of(store: Store, cells) -> np.ndarray:
    """The slots of own cells, by their numbers."""
    cells = np.asarray(cells)
    return store.tile_slots.numpy()[store.tile_home.numpy()[cells // TILE] * TILE + cells % TILE]


def joints_of(store: Store) -> tuple:
    """The live joints' slots, the slots of the volume before and the volume after each, and their axes."""
    own = store.own_cell.numpy()[:store.count]
    joints = np.flatnonzero(store.cell_quanta.numpy()[own] < 0)
    axis = np.argmax(steps_of(own[joints], store.side) % 2, axis=1)
    ends = [slots_of(store, own[joints] + way * store.side ** (2 - axis)) for way in (-1, 1)]
    return joints, ends, axis
