"""Cut: what a tear does to the torn volume's joints, and how a tear's new cells are made.

A joint on the torn volume's far side along the tear's direction goes with the far half. A
joint along its side lies on a part of that side: within the near or the far half, it goes
with that half and its part is that half's; across the whole, it is cut in two with the
tear, one part to each half, each keeping the minus quanta of its own half of the joint's
fluid and half of the part of the other volume's side. So every joint lies between two.
"""
import numpy as np

from physics.engine.axiom_3_structure.definitions.relations import joints_of
from physics.engine.axiom_3_structure.definitions.store import Store, put, slot_of
from physics.engine.axiom_3_structure.logic.take_own import lay, take_own
from physics.engine.axiom_6_limits.definitions.overstretched import layers


def relay(store: Store, cell: int, quanta: int) -> None:
    """Give an own cell a new total and lay its boxes as a whole ball whirl of it."""
    put(store.cell_quanta, cell, quanta)
    put(store.cell_born, cell, store.step)
    lay(store, store.own_cell, slot_of(store, cell), 1, store.box_quanta, store.box_phase)


def new_cell(store: Store, quanta: int, laid: int) -> int:
    """The next cell not in use, made now with its own boxes: a ball whirl of `quanta`."""
    if store.cells == store.cell_quanta.shape[0]:
        raise RuntimeError(f"no room for another cell: {store.cells} are in use")
    cell, store.cells = store.cells, store.cells + 1
    put(store.cell_laid, cell, laid)
    take_own(store, cell)
    relay(store, cell, quanta)
    return cell


def pass_on(store: Store, volume: int, other: int, way: int) -> None:
    """Hand the joints of a volume torn across `way` to its halves: `volume` near, `other` far."""
    ends, parts = store.joint_ends.numpy(), store.joint_part.numpy()
    for joint, end in joints_of(ends[:store.cells], volume):
        at = 2 * ((way - ends[joint, 2] - 1) % 3)  # where the part's numbers along `way` are
        if ends[joint, 2] == way:
            ends[joint, end] = volume if end else other  # named first: it lay on the far side
        elif parts[joint, end, at + 1] < 1.0:  # within one half: that half's, twice as wide there
            far = parts[joint, end, at] >= 0.5
            parts[joint, end, at:at + 2] = 2 * parts[joint, end, at] - far, 2 * parts[joint, end, at + 1]
            ends[joint, end] = other if far else volume
        else:
            block = store.box_quanta[slot_of(store, joint)].numpy().reshape((store.across,) * 3)
            near, far = np.split(layers(block, way), 2)
            twin = new_cell(store, -int(far.sum()), int(store.cell_laid.numpy()[joint]))
            relay(store, joint, -int(near.sum()))
            parts[joint, 1 - end, at + 1] /= 2  # each keeps half the part of the other volume's side
            ends[twin], parts[twin] = ends[joint], parts[joint]
            ends[twin, end], parts[twin, 1 - end, at] = other, parts[joint, 1 - end, at:at + 2].sum()
            store.cuts += 1
    store.joint_ends.assign(ends)
    store.joint_part.assign(parts)
