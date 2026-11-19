"""Tear: where enough leaked energy has gathered at a free place, the leak has torn a volume off.

Leaked quanta gather at a free place as its total (`leak.py`); they are no cell yet and do
nothing. Once every part of them could be a whirl (`PARTS` times the least whirl) the
place is a new energy volume. What gathered is shared in seven equal parts, the volume's
own and one toward each of its six sides. Toward a side where a volume stands, or forms in
the same step, that part flips from plus to minus and is the joint between the two: the
place that tore, holding the least it ever will. Toward a side with no volume the part
stays in the new volume. Two volumes that form side by side both give their part to the
joint between them. Plus lost is minus gained, to the quantum. Every new cell is laid as a
whole ball whirl, by its place (`laid_at`). Found on the host, before the fluid moves.
"""
import numpy as np

from physics.engine.axiom_2_swirls.definitions.swirl_pattern import laid_at
from physics.engine.axiom_3_structure.definitions.places import steps_of
from physics.engine.axiom_3_structure.definitions.store import Store
from physics.engine.axiom_3_structure.logic.take_own import take_all
from physics.engine.axiom_6_limits.definitions.least_whirl import PARTS, least_whirl


def torn(store: Store) -> dict:
    """The cells the leak tears off now, each with its quanta: new volumes plus, their joints minus."""
    held = store.cell_quanta.numpy().astype(np.int64)
    stands = np.zeros(held.size, dtype=bool)
    stands[store.own_cell.numpy()[:store.count]] = True
    ripe = np.flatnonzero((held >= PARTS * least_whirl(store.across)) & ~stands)
    if ripe.size and not (1 < steps_of(ripe, store.side).min()
                          and steps_of(ripe, store.side).max() < store.side - 2):
        raise RuntimeError(f"no room for another layer: the store holds {store.side // 4}")
    stands[ripe], made = True, {}
    for cell in ripe.tolist():
        sides = [cell + way * store.side ** axis for axis in range(3) for way in (-1, 1)]
        joints = [side for side in sides if stands[2 * side - cell]]
        for joint in joints:
            made[joint] = made.get(joint, 0) - int(held[cell]) // PARTS
        made[cell] = int(held[cell]) - len(joints) * (int(held[cell]) // PARTS)
    return made


def tear(store: Store) -> None:
    """Make every cell the leak tears off in this step, each with its own boxes."""
    made = dict(sorted(torn(store).items())) if store.leaks else {}
    if made:
        held, laid, cells = store.cell_quanta.numpy(), store.cell_laid.numpy(), list(made)
        held[cells] = list(made.values())
        laid[cells] = [laid_at(steps) for steps in steps_of(cells, store.side)]
        store.cell_quanta.assign(held)
        store.cell_laid.assign(laid)
        take_all(store, cells)
