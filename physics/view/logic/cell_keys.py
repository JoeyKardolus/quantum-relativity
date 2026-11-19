"""Cell keys: what the one-cell window's keys do to its cell, with exact books.

`scaled` doubles or halves the cell's energy box for box (a halved odd box keeps the
smaller half; the cell's total is the boxes' sum), so the ball is half or twice as wide by
the size law and its whirl turns twice or half as fast. `flipped` turns it the other way.
A calm cell takes its own boxes first. `going` sets the crossing's fuller volume moving
toward the other. Staging by hand, as a scene's start is; no step.
"""
import numpy as np

from physics.engine.axiom_2_swirls.logic.flip import flip
from physics.engine.axiom_3_structure.definitions.store import Store, put, slot_of
from physics.engine.axiom_3_structure.logic.take_own import take_own
from physics.engine.axiom_5_packing.logic.lumps import moving

MOST, LEAST = 2 ** 29, 216_000  # no doubling past half a whole number, no halving below 1,000 a box at 6


def scaled(store: Store, cell: int, more: bool) -> None:
    """Double (`more`) or halve the energy of the cell, every box alike."""
    take_own(store, cell)
    slot, boxes = slot_of(store, cell), store.box_quanta.numpy()
    held = np.abs(boxes[slot].astype(np.int64)).sum()
    if (more and held > MOST) or (not more and held < LEAST):
        return
    boxes[slot] = boxes[slot] * 2 if more else np.sign(boxes[slot]) * (np.abs(boxes[slot]) // 2)
    store.box_quanta.assign(boxes)
    put(store.cell_quanta, cell, int(boxes[slot].sum()))


def flipped(store: Store, cell: int) -> None:
    """Turn the cell's whirl the other way: plus becomes minus, red blue."""
    take_own(store, cell)
    flip(store, cell)


def going(store: Store) -> None:
    """Set the crossing's fluid going along the line: the first volume's toward the joint, and
    the phase falling on across it, so the second volume stands a step behind the cap it faces."""
    moving(store, [0], 1, 0.3)
    phase = store.box_phase.numpy()
    phase[2] -= 0.3 * (store.across // 2 + 1)
    store.box_phase.assign(((phase + np.pi) % (2 * np.pi) - np.pi).astype(np.float32))
