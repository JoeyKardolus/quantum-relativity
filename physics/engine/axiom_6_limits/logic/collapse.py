"""Collapse: a joint whose two volumes are both past the limit flips to plus where it stands.

Found on the host, after the tear and before the fluid moves, from the pull of the step
before (`past_limit`): where nothing at the top speed could climb out on either side of a
joint, it has nothing left to hold apart. Its minus quanta flip to plus, the same amount,
laid as whole ring whirls turning a volume's way, and from then on it and its two volumes
are one volume: plus beside plus with no joint between. Nothing else changes. Every cell
keeps its place, its own whole number and its other joints, so no number grows at a merge
(the merged volume's energy is its cells' sum) and cells far away notice only the deeper
pull, handed on a cell a step.
"""
import numpy as np

from physics.engine.axiom_3_structure.definitions.store import Store
from physics.engine.axiom_3_structure.logic.joints_of import joints_of
from physics.engine.axiom_3_structure.logic.take_own import lay
from physics.engine.axiom_6_limits.definitions.collapse_limit import past_limit


def past(store: Store) -> np.ndarray:
    """The slots of the live joints whose two volumes are both past the limit now."""
    pull = store.cell_pull.numpy()[store.own_cell.numpy()[:store.count]]
    if not past_limit(pull).any():
        return np.zeros(0, dtype=int)
    joints, (first, second), _axis = joints_of(store)
    return joints[past_limit(np.minimum(pull[first], pull[second]))]


def collapse(store: Store) -> None:
    """Every joint past the limit flips to plus where it stands; the books stay exact."""
    gives = past(store) if store.side else np.zeros(0, dtype=int)
    if gives.size == 0:
        return
    quanta, cells = store.cell_quanta.numpy(), store.own_cell.numpy()[gives]
    store.collapses += int(gives.size)
    store.flipped -= int(quanta[cells].astype(np.int64).sum())
    quanta[cells] *= -1
    store.cell_quanta.assign(quanta)
    for slot in gives.tolist():  # each laid as a volume's whirl of what it holds
        lay(store, store.own_cell, slot, 1, store.box_quanta, store.box_phase)
