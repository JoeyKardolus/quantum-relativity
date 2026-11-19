"""Meshing: whether the fluid of two cells moves alike where they face, read off their boxes.

For each row where the cell `before` faces the cell `after` along `axis`: how far apart
their two phases are, less what each has turned past the calm turn (`own_turned`), and
which way each cap's fluid moves along its ring's roll: out from the jet, or in toward it.
A volume and the joint beside it turn opposite ways, so their caps agree and both move
out; two volumes laid side by side with no joint between meet half a turn apart, one
moving out and one in: they grind.
"""
import numpy as np

from physics.engine.axiom_2_swirls.definitions.ring_boxes import cap_boxes
from physics.engine.axiom_3_structure.definitions.store import Store, slot_of


def meshing(store: Store, before: int, after: int, axis: int) -> tuple[float, float]:
    """(agreement, alike): the mean cosine of the facing phases' difference (1: agree, -1: half
    a turn apart) and the share of rows where both caps' fluid moves the same way."""
    across, phase, turned = store.across, store.box_phase.numpy(), store.own_turned.numpy()
    first, second = slot_of(store, before), slot_of(store, after)
    front, back = np.array(cap_boxes(across, axis, 1)), np.array(cap_boxes(across, axis, 0))
    apart = (phase[first, front] - turned[first]) - (phase[second, back] - turned[second])
    out_front = np.sin(phase[first, front] - phase[first, front + across])   # on, round the core
    out_back = np.sin(phase[second, back] - phase[second, back - across])    # back, round the core
    return float(np.cos(apart).mean()), float(np.mean(np.sign(out_front) == np.sign(out_back)))
