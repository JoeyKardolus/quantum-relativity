"""Direction codes: the 26 directions a neighbour can lie in, as whole numbers.

Contract: a direction is a step (dx, dy, dz), each -1, 0 or +1 and not all 0. Its code is
`(dx + 1) * 9 + (dy + 1) * 3 + dz + 1`, so 0 to 26 without `CENTRE` (13, the step
(0, 0, 0), which is no direction). A code is a label of a relation between two cells
(axiom 3: a cell has no geometry of its own), never a place.
  opposite(code)          Warp func: the code of the reversed step, 26 - code
  step_of(code)           Warp func: the step as a vec3i
  opposite_host(code)     host twin
  step_of_host(code)      host twin, a tuple (dx, dy, dz)
  code_of_host(dx, dy, dz)  the code of a step; raises ValueError for a step outside -1 .. 1
"""
from __future__ import annotations

import warp as wp

wp.set_module_options({"fuse_fp": False})

CENTRE = 13


@wp.func
def opposite(code: wp.int32) -> wp.int32:
    """The code of the direction pointing the other way."""
    return 26 - code


@wp.func
def step_of(code: wp.int32) -> wp.vec3i:
    """The step (dx, dy, dz) a code stands for."""
    return wp.vec3i(code // 9 - 1, (code // 3) % 3 - 1, code % 3 - 1)


def opposite_host(code: int) -> int:
    """The code of the direction pointing the other way (host)."""
    return 26 - int(code)


def step_of_host(code: int) -> tuple:
    """The step (dx, dy, dz) a code stands for (host)."""
    code = int(code)
    return code // 9 - 1, (code // 3) % 3 - 1, code % 3 - 1


def code_of_host(dx: int, dy: int, dz: int) -> int:
    """The code of the step (dx, dy, dz)."""
    if not all(part in (-1, 0, 1) for part in (dx, dy, dz)):
        raise ValueError(f"a step is -1, 0 or +1 along each axis, got {(dx, dy, dz)}")
    return (int(dx) + 1) * 9 + (int(dy) + 1) * 3 + int(dz) + 1
