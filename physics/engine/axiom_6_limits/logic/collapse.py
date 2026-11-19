"""Collapse: a joint past its limit flips to plus and merges with its two volumes into one.

Found on the host, after the tear and before the fluid moves, from the pull of the step
before and the distances as they are now (`past_limit`). The joint's minus quanta flip to plus, the
same amount, and the joint and its two volumes are one volume that holds all three: it
stands at the place of the fuller of the two, laid as whole ring whirls, and by the
size law it is smaller and pulls harder, so its other joints are squeezed more and may
follow in the steps after. It keeps every joint its two volumes had (`shared_sides.py`);
a joint that so comes to lie inside one volume has nothing to hold apart and flips into
it. The places of the joint and of the volume that went hold nothing from then on.
"""
import numpy as np

from physics.engine.axiom_3_structure.definitions.store import Store
from physics.engine.axiom_3_structure.logic.shared_sides import joints_of, share_sides
from physics.engine.axiom_3_structure.logic.take_own import lay
from physics.engine.axiom_5_packing.logic.squeeze import squeeze
from physics.engine.axiom_6_limits.definitions.collapse_limit import past_limit

FULL = 2 ** 31  # a cell's whole number holds less than this


def _fall(held: np.ndarray, whole: np.ndarray, ring: np.ndarray, joint: int, ends: tuple) -> int:
    """The joint in slot `joint` flips into the volumes of its `ends`, made one; the slot that one stands in."""
    volumes = sorted({int(whole[ends[0]]), int(whole[ends[1]])}, key=lambda slot: (-held[slot], slot))
    keep, total = volumes[0], int(held[volumes].sum() - held[joint])
    if total >= FULL:
        raise RuntimeError(f"a cell's whole number is full: a merged volume would hold {total:,} quanta")
    for gone in volumes[1:]:
        part = gone
        while True:  # every slot of the volume that goes is part of the one that stands
            whole[part], part = keep, ring[part]
            if part == gone:
                break
        ring[keep], ring[gone] = ring[gone], ring[keep]
    held[volumes], held[joint] = 0, 0
    held[keep] = total
    return keep


def past(store: Store) -> np.ndarray:
    """The slots of the live joints that are past their limit now."""
    squeeze(store)  # the distances as they are, whoever looked in between
    own = store.own_cell.numpy()[:store.count]
    apart = store.own_squeeze.numpy()[:store.count, 2]
    deep = past_limit(store.cell_pull.numpy()[own], apart, store.turn_per_step / (2 * np.pi))
    return np.flatnonzero(deep & (store.cell_quanta.numpy()[own] < 0))


def collapse(store: Store) -> None:
    """Every joint past its limit flips to plus and merges with its two volumes; the books stay exact."""
    gives = past(store) if store.side else np.zeros(0, dtype=int)
    if gives.size == 0 and store.collapses == 0:
        return
    if gives.size:
        own, quanta = store.own_cell.numpy()[:store.count], store.cell_quanta.numpy()
        before, held = quanta[own], quanta[own].astype(np.int64)
        whole, ring = store.own_whole.numpy(), store.own_ring.numpy()
        joints, (first, second), _axis = joints_of(store)
        ends = dict(zip(joints.tolist(), zip(first.tolist(), second.tolist())))
        stands = [_fall(held, whole, ring, joint, ends[joint]) for joint in gives.tolist()]
        stands += [_fall(held, whole, ring, joint, pair) for joint, pair in ends.items()
                   if held[joint] < 0 and whole[pair[0]] == whole[pair[1]]]  # now inside one volume
        fell = (before < 0) & (held == 0)
        store.collapses += int(fell.sum())
        store.flipped -= int(before[fell].sum())
        quanta[own] = held
        for kept, now in ((store.cell_quanta, quanta), (store.own_whole, whole), (store.own_ring, ring)):
            kept.assign(now)
        for slot in np.flatnonzero((before != 0) & (held == 0)).tolist():
            store.box_quanta[slot].zero_()
        for slot in sorted(set(stands)):
            lay(store, store.own_cell, slot, 1, store.box_quanta, store.box_phase)
    share_sides(store)  # new joints at a merged volume's side share it too
