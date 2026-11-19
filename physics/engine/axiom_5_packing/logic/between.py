"""Between: which cells are volumes, and the way from each volume to each other through the cells' sizes.

A cell has no place in space, only a size and neighbours. Along one direction the way from
a volume to another is the sizes of the cells between them in a row, from middle to middle:
half the one, every cell between, half the other (for two neighbours: `apart_of`). The
two volumes need not lie in one row, so each direction is counted twice, along the row of
the one and along the row of the other, and the mean is taken: the way back is the same way
turned round. The distance is the length of the three together. A place no cell stands at
has no size and adds none. Read on the host.
"""
import numpy as np

from physics.engine.axiom_3_structure.definitions.kinds import QUANTA_A_BOX
from physics.engine.axiom_3_structure.definitions.places import steps_of
from physics.engine.axiom_3_structure.definitions.store import Store


def held_of(store: Store) -> np.ndarray:
    """Every place's quanta in calm volumes, signed, as a block of places; 0 where no own cell stands."""
    own, held = store.own_cell.numpy()[:store.count], np.zeros(store.side ** 3)
    held[own] = store.cell_quanta.numpy()[own] / (QUANTA_A_BOX * store.across ** 3)
    return held.reshape((store.side,) * 3)


def volumes_of(store: Store, held: np.ndarray) -> np.ndarray:
    """The own energy volumes: plus quanta at a volume's place (every step even)."""
    cells = np.flatnonzero(held.ravel() > 0)
    return cells[(steps_of(cells, store.side) % 2 == 0).all(axis=1)]


def between(held: np.ndarray, at: np.ndarray) -> np.ndarray:
    """The way from each volume to each other, (volumes, volumes, 3) in calm lengths; `at` their steps."""
    size = np.divide(1.0, np.abs(held), out=np.zeros_like(held), where=held != 0)
    count = len(at)
    way = np.empty((count, count, 3))
    for axis in range(3):
        middle = np.cumsum(size, axis=axis) - size / 2
        row = [np.broadcast_to(at[:, None, one], (count, count)) for one in range(3)]
        row[axis] = np.broadcast_to(at[None, :, axis], (count, count))  # the other's step, in the own row
        along = middle[tuple(row)] - middle[tuple(at.T)][:, None]
        way[..., axis] = (along - along.T) / 2
    return way
