"""Torn: the tear's law read on the host, place by place, for a check to hold the card's tear against.

Leaked quanta gather at a free place as its total; once every part of them could be a
whirl (`PARTS` times the least whirl) the place is a new energy volume. What gathered is
shared in seven equal parts, the volume's own and one toward each of its six sides. Toward
a side where a volume stands, or forms in the same step, that part flips from plus to
minus and is the joint between the two. Toward a side with no volume the part stays.
`tear.py` does the same on the card; no step calls this.
"""
import numpy as np

from physics.engine.axiom_3_structure.definitions.places import steps_of
from physics.engine.axiom_3_structure.definitions.store import Store
from physics.engine.axiom_6_limits.definitions.least_whirl import PARTS, least_whirl


def torn(store: Store) -> dict:
    """The cells the leak tears off now, each with its quanta: new volumes plus, their joints minus."""
    held = store.cell_quanta.numpy().astype(np.int64)
    stands = np.zeros(held.size, dtype=bool)
    stands[store.own_cell.numpy()[:store.count]] = True
    ripe = np.flatnonzero((held >= PARTS * least_whirl(store.across)) & ~stands)
    stands[ripe], made = True, {}
    for cell in ripe.tolist():
        sides = [cell + way * store.side ** axis for axis in range(3) for way in (-1, 1)]
        joints = [side for side in sides if stands[2 * side - cell]]
        for joint in joints:
            made[joint] = made.get(joint, 0) - int(held[cell]) // PARTS
        made[cell] = int(held[cell]) - len(joints) * (int(held[cell]) // PARTS)
    return made
