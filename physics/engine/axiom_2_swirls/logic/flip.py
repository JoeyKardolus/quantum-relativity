"""Flip: turn an own cell's whirl the other way. The sign is the direction of the turn.

A scene's act, never a step's: the cell's quanta change sign and its fluid becomes its own
mirror image, the boxes round each core in the opposite order. Nothing else changes: each
ring holds what it held, the cell is as big and turns as fast, the other way round.
"""
from physics.engine.axiom_3_structure.definitions.store import Store, peek, put, slot_of


def flip(store: Store, cell: int) -> None:
    """Change the sign of the own cell `cell` and mirror its fluid round its three cores."""
    slot, across = slot_of(store, cell), store.across
    quanta, phase = store.box_quanta.numpy(), store.box_phase.numpy()
    for boxes in (quanta, phase):
        boxes[slot] = boxes[slot].reshape(across, across, across)[:, ::-1, :].ravel()
    quanta[slot] *= -1
    store.box_quanta.assign(quanta)
    store.box_phase.assign(phase)
    put(store.cell_quanta, cell, -peek(store.cell_quanta, cell))
