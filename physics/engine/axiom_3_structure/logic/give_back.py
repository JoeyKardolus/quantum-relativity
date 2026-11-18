"""Give back: an own cell that is calm again becomes a rule again and frees its slot.

It may go when its fluid still winds the swirl's whole number on every ring and
`off_rule` is under 1. No quantum is lost or made; what is dropped is the shape of its
difference from the rule, and the quanta that so change box are added to the ledger.
Layer 1 does this for one cell from the host; many cells on the card come with layer 2.
"""
import numpy as np
import warp as wp

from physics.engine.axiom_3_structure.definitions.store import Store, entry_of, peek, put, slot_of
from physics.engine.axiom_3_structure.logic.boxes_of import rule_boxes, swirl_of

CALM_UNDER = 1.0  # `off_rule` of plain counting noise; fixed, never moved to fit a run


def give_back(store: Store, cell: int) -> bool:
    """Send an own cell back to calm if it may go; says whether it went."""
    slot, last, seen = slot_of(store, cell), store.count - 1, swirl_of(store, cell)
    laid_turns = 2 * (peek(store.cell_laid, cell) % 2) - 1  # the phase falls the way the fluid goes
    if slot < 0 or seen["turns"] != laid_turns or seen["off_rule"] >= CALM_UNDER:
        return False
    held = store.box_quanta[slot].numpy().astype(np.int64)
    put(store.cell_quanta, cell, int(held.sum()))
    store.ledger += int(np.abs(held - rule_boxes(store, cell)[0][0].numpy()).sum()) // 2
    if slot != last:
        moving = peek(store.own_cell, last)  # the last slot's cell moves into the hole
        wp.copy(store.box_quanta[slot], store.box_quanta[last])
        wp.copy(store.box_phase[slot], store.box_phase[last])
        put(store.own_cell, slot, moving)
        put(store.tile_slots, entry_of(store, moving), slot)
    put(store.tile_slots, entry_of(store, cell), -1)
    store.count = last
    return True
