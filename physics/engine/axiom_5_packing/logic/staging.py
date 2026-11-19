"""Staging: two acts for a check or a scene's start, never for a step.

`copy_of` gives a store again with every array its own, so several checks can each stage
their own in one grown patch. `packed` makes chosen volumes hold a set number of quanta,
each laid as a whole whirl of it.
"""
from dataclasses import fields, replace

import numpy as np
import warp as wp

from physics.engine.axiom_3_structure.definitions.store import Store
from physics.engine.axiom_3_structure.logic.take_own import lay


def copy_of(store: Store) -> Store:
    """The store again, each array its own."""
    return replace(store, **{one.name: wp.clone(getattr(store, one.name))
                             for one in fields(store) if isinstance(getattr(store, one.name), wp.array)})


def packed(store: Store, cells: list, quanta: int) -> None:
    """Pack each of the own volumes `cells` to hold `quanta`, laid as a whole whirl."""
    held, own = store.cell_quanta.numpy(), store.own_cell.numpy()[:store.count]
    held[cells] = quanta
    store.cell_quanta.assign(held)
    for slot in np.flatnonzero(np.isin(own, cells)).tolist():
        lay(store, store.own_cell, slot, 1, store.box_quanta, store.box_phase)
