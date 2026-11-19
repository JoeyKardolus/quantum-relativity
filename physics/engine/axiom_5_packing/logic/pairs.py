"""Pairs: the pull's law read on the host, pair by pair, for a check to hold the card's pull against.

All pairs, in numpy (`definitions/pull.py` has the law). The volumes' energy is written
into the past kept (`cell_past`, a ring of `REACH` steps, a volume's place a column); a
collapsed joint's plus counts half to each of the two volumes it merged. Then for each
pair: the way between (`between`), how late the pull is (`steps_late`), the other's energy
then. A volume is in no sum of its own. Before a pull has had the time to arrive there is
none. `pull.py` does the same on the card; no step calls this.
"""
import numpy as np
import warp as wp

from physics.engine.axiom_3_structure.definitions.places import steps_of
from physics.engine.axiom_3_structure.definitions.store import Store
from physics.engine.axiom_5_packing.definitions.groups import nodes_of, top_of, wide_of
from physics.engine.axiom_5_packing.definitions.pull import REACH, steps_late
from physics.engine.axiom_5_packing.logic.between import between, held_of, volumes_of


def _energy(held: np.ndarray, at: np.ndarray) -> np.ndarray:
    """Each volume's pulling energy: its own plus, and half of each collapsed joint's at its sides."""
    plus = np.pad(held.clip(0), 1)
    sides = sum(plus[tuple((at + 1 + way * np.eye(3, dtype=int)[axis]).T)] for axis in range(3) for way in (-1, 1))
    return plus[tuple((at + 1).T)] + sides / 2


def past_of(store: Store) -> wp.array:
    """The store's past, made empty the first time: (`REACH`, volumes' places and groups), on its device."""
    if store.cell_past is None:
        nodes = nodes_of(store.side, top_of(store.side, store.near))
        store.cell_past = wp.zeros((REACH, nodes), dtype=wp.float32, device=store.box_quanta.device)
    return store.cell_past


def pulls(store: Store) -> tuple:
    """The own volumes, how deep each lies and the pull on each along the three directions (a speed a turn)."""
    held, a_turn = held_of(store), store.turn_per_step / (2.0 * np.pi)
    cells = volumes_of(store, held)
    at = steps_of(cells, store.side)
    column = np.ravel_multi_index(tuple((at // 2).T), (wide_of(store.side),) * 3)
    past, energy = past_of(store).numpy(), _energy(held, at)
    past[store.step % REACH, :wide_of(store.side) ** 3] = 0.0
    past[store.step % REACH, column] = energy
    store.cell_past.assign(past)
    way = between(held, at)
    apart = np.linalg.norm(way, axis=2)
    np.fill_diagonal(apart, np.inf)  # no volume is another to itself
    late = steps_late(apart, a_turn)
    then = np.where(late <= store.step, past[(store.step - late) % REACH, column[None, :]], 0.0)
    return cells, energy + (then / apart).sum(axis=1), ((then / apart ** 3)[..., None] * way).sum(axis=1)
