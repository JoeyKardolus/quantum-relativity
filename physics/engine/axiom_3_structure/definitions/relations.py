"""Relations: which cells lie next to which. Nothing is laid ready-made; only a tear adds one.

A joint knows its two volumes and the direction it lies along (`joint_ends`: the volume
before it, the one after, the axis): it lies on the far side of the first along that axis
and on the near side of the second. A volume has six sides; a tear's halves keep their
parent's frame, so every volume's directions agree. A volume's joints are those that name it.

`joint_part` says, for each of its two volumes, which part of that side the joint lies on:
along the two directions after its axis, where the part starts and how wide it is, the
whole side being 0 to 1. A joint made by a tear has the whole side; a cut halves the part.
The boxes of a volume that face a joint are those whose middles lie in its part.
"""
import numpy as np
import warp as wp

from physics.engine.axiom_3_structure.definitions.store import Store

WHOLE = (0.0, 1.0, 0.0, 1.0)  # a part that is the whole side


@wp.func
def box_along(start: float, wide: float, row: wp.int32, across: wp.int32) -> wp.int32:
    """The box along one direction that a joint's row `row` faces in a part; -1: none lies in it."""
    first = int(wp.ceil(start * float(across) - 0.5))
    boxes = int(wp.ceil((start + wide) * float(across) - 0.5)) - first
    if boxes <= 0:
        return -1
    return first + row * boxes // across


def join(store: Store, joint: int, before: int, after: int, axis: int) -> None:
    """Make cell `joint` the joint between two volumes along `axis`, on the whole of both sides."""
    store.joint_ends[joint:joint + 1].fill_(wp.vec3i(before, after, axis))
    store.joint_part[joint].fill_(wp.vec4(*WHOLE))


def joints_of(ends: np.ndarray, volume: int) -> list[tuple[int, int]]:
    """A volume's joints, read off `joint_ends` on the host: each joint and which end names it."""
    return [(int(joint), int(end)) for joint, end in zip(*np.nonzero(ends[:, :2] == volume))]
