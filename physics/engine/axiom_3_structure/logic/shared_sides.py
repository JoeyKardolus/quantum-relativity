"""Shared sides: which volumes each live joint lies between now, and how joints share a volume's side.

A merged volume keeps the joints its volumes had, so several may lie on one of its six
sides. `share_sides` gives each joint its rank among the joints on that side and their
count, for each of its two volumes (`own_part`): crossing takes the side's rows in turn and
the pull takes their mean. Two joints between the same two volumes both stay, each with
its part of the side.
"""
import numpy as np

from physics.engine.axiom_3_structure.definitions.merged import ALONE, slots_of
from physics.engine.axiom_3_structure.definitions.places import steps_of
from physics.engine.axiom_3_structure.definitions.store import Store


def joints_of(store: Store) -> tuple:
    """The live joints' slots, the slots of the volume before and the volume after each, and their axes."""
    own = store.own_cell.numpy()[:store.count]
    joints = np.flatnonzero(store.cell_quanta.numpy()[own] < 0)
    axis = np.argmax(steps_of(own[joints], store.side) % 2, axis=1)
    whole = store.own_whole.numpy()
    ends = [whole[slots_of(store, own[joints] + way * store.side ** (2 - axis))] for way in (-1, 1)]
    return joints, ends, axis


def share_sides(store: Store) -> None:
    """Write each live joint's rank and count on the side of its volume before and of its volume after."""
    joints, ends, axis = joints_of(store)
    part = np.tile(ALONE, (store.own_part.shape[0], 1))
    for end in range(2):
        side = ends[end].astype(np.int64) * 3 + axis  # one side of one volume
        order = np.lexsort((joints, side))
        _, first, count = np.unique(side[order], return_index=True, return_counts=True)
        part[joints[order], 2 * end] = np.arange(len(order)) - np.repeat(first, count)
        part[joints[order], 2 * end + 1] = np.repeat(count, count)
    store.own_part.assign(part)
