"""Jiggle: how hard the energy fluid jiggles inside each energy volume, per quantum, its heat.

Contract: `jiggle(state)` sets state.jiggle for every energy volume. Over every face
between two boxes of the cell's own block, with n and m their quanta and d the phase
difference across the face brought into (-pi, pi], weighted w = (n + m) / 2: per axis
the weighted spread of d about the cell's mean along that axis, sum(w d ** 2) -
sum(w d) ** 2 / sum(w), summed over the three axes and divided by the summed weights.
With hbar = box mass = 1 the fluid's speed along an axis is its phase difference per
box, so this is the quanta-weighted mean square of the speed's undirected part along
one axis: the energy of the fluid's own jiggling per quantum, its heat (friction turns
directed flow into jiggling; pour.py raises it where energy arrives). Each cell is
summed by one thread in a fixed order. Joints get nothing. Returns None.
"""
from __future__ import annotations

import warp as wp

from physics.engine.state import BLOCK, State

wp.set_module_options({"fuse_fp": False})

PI = 3.14159265358979


@wp.kernel
def _jiggle(kind: wp.array(dtype=wp.int32), quanta: wp.array4d(dtype=wp.int32),
            phase: wp.array4d(dtype=wp.float32), heat: wp.array(dtype=wp.float32), across: int):
    cell = wp.tid()
    if kind[cell] != 1:
        return
    spread = wp.float64(0.0)
    weight = wp.float64(0.0)
    for axis in range(3):
        square = wp.float64(0.0)
        first = wp.float64(0.0)
        total = wp.float64(0.0)
        step = wp.vec3i(0, 0, 0)
        step[axis] = 1
        for one in range(across):
            for two in range(across):
                for three in range(across):
                    here = wp.vec3i(one, two, three)
                    if here[axis] == across - 1:
                        continue
                    there = here + step
                    share = wp.float64(0.5 * float(quanta[cell, one, two, three]
                                                   + quanta[cell, there[0], there[1], there[2]]))
                    turn = phase[cell, there[0], there[1], there[2]] - phase[cell, one, two, three]
                    turn = turn - 2.0 * PI * wp.floor((turn + PI) / (2.0 * PI))
                    square += share * wp.float64(turn * turn)
                    first += share * wp.float64(turn)
                    total += share
        spread += square - first * first / total
        weight += total
    heat[cell] = wp.float32(spread / weight)


def jiggle(state: State) -> None:
    """Measure each energy volume's own heat per quantum from its boxes."""
    wp.launch(_jiggle, dim=state.count, device=state.device,
              inputs=[state.kind, state.quanta, state.phase, state.jiggle, BLOCK])
