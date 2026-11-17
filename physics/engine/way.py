"""Way (Warp func): one of the pairs of energy volumes a joint lies between.

Contract: `way(joint, index, side, way_at, side_n, ways_n, way_pool)` returns the two
volumes of way `index` (0 .. ways_n[joint] - 1) of `joint` as a vec2i. A joint that
keeps its own sides (state.py) has side_n volumes on its first side and ways_n / side_n
on its second, listed one after the other from way_pool[way_at[joint]]; way `index` is
the pair (first side's volume index // second count, second side's volume index %
second count). A pristine face joint's one way is the two lattice places either side
of it along its one odd index. The two may be the same volume (a joint shut inside
one volume): callers skip such a pair, it is no way. Energy crosses and volumes press
only along a way; edge and corner joints of the lattice have none.
"""
from __future__ import annotations

import warp as wp

from physics.engine.lattice_step import lattice_step


@wp.func
def way(joint: int, index: int, side: int, way_at: wp.array(dtype=wp.int32), side_n: wp.array(dtype=wp.int32),
        ways_n: wp.array(dtype=wp.int32), way_pool: wp.array(dtype=wp.int32)) -> wp.vec2i:
    """The two energy volumes at the ends of the joint's way number `index`."""
    at = way_at[joint]
    if at >= 0:
        near = side_n[joint]
        far = ways_n[joint] // near
        return wp.vec2i(way_pool[at + index // far], way_pool[at + near + index % far])
    reach = 9
    if (joint // side) % 2 == 1:
        reach = 3
    if joint % 2 == 1:
        reach = 1
    return wp.vec2i(lattice_step(joint, side, 13 - reach), lattice_step(joint, side, 13 + reach))
