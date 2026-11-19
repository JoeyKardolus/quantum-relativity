"""Boxes of: a cell's boxes whichever way it is kept, for reading and for drawing.

A calm cell has none, so the rule's boxes are dealt into one spare slot, as `take_own` deals them.
"""
import warp as wp

from physics.engine.axiom_2_swirls.logic.read_swirl import (
    off_rule, round_flow, still_share, whole_turns)
from physics.engine.axiom_3_structure.definitions.store import Store, slot_of
from physics.engine.axiom_3_structure.logic.take_own import lay


def rule_boxes(store: Store, cell: int) -> tuple[wp.array, wp.array]:
    """The boxes the rule gives the cell at this step: quanta and phase, in slot 0."""
    with wp.ScopedDevice(store.box_quanta.device):
        quanta = wp.zeros((1, store.across ** 3), dtype=wp.int32)
        phase = wp.zeros((1, store.across ** 3), dtype=wp.float32)
        lay(store, wp.array([cell], dtype=wp.int32), 0, 1, quanta, phase)
    return quanta, phase


def boxes_of(store: Store, cell: int) -> tuple[wp.array, wp.array, int]:
    """The cell's boxes on the device: quanta, phase, and the slot to look in."""
    slot = slot_of(store, cell)
    return (store.box_quanta, store.box_phase, slot) if slot >= 0 else (*rule_boxes(store, cell), 0)


def swirl_of(store: Store, cell: int) -> dict:
    """What the cell's fluid is doing now, read on the host: `read_swirl`'s measures, each ring's."""
    box_quanta, box_phase, slot = boxes_of(store, cell)
    quanta, phase = box_quanta[slot].numpy(), box_phase[slot].numpy()
    rule = rule_boxes(store, cell)[0][0].numpy()
    return {"turns": tuple(whole_turns(phase, ring) for ring in range(3)),
            "round_flow": tuple(round_flow(quanta, phase, ring) for ring in range(3)),
            "still": still_share(quanta, phase),
            "off_rule": off_rule(quanta, rule)}
