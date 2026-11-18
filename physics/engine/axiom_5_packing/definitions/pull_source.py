"""Pull source: more packed pulls harder, and only plus pulls.

Contract: `pull_source(kind, cell_quanta, reach, pull_strength)` (Warp func; int32,
int64, float64, float64) returns `4 pi G * cell_quanta / reach ** 3` as float64 where
the sign law says the kind makes a pull (an energy volume), else 0: a joint carries the
pull but adds none. `reach` is the cell's mean distance to the neighbours it hears
(heard_pull.py): the room it stands in. So the source is how packed the cell is in that
room; with the wave law (pull_wave_law.py) what leaves a cell is `4 pi G` times the
energy it holds, a cell that holds more or stands in less room sits deeper, and what a
cell makes of the pull goes as what it feels of it: two lumps pull each other equally
hard and the pair gains no motion out of nothing. 0 for an empty cell or an endless
reach. (Until 2026-11-18 the source was the cell's own packing, the quanta to the fourth
power: a lump of twice the energy then pulled sixteen times harder than it was pulled.)
"""
from __future__ import annotations

import math

import warp as wp

from physics.engine.axiom_2_swirls.definitions.sign_law import makes_pull

wp.set_module_options({"fuse_fp": False})

_FOUR_PI = wp.constant(wp.float64(4.0 * math.pi))


@wp.func
def pull_source(kind: wp.int32, cell_quanta: wp.int64, reach: wp.float64, pull_strength: wp.float64) -> wp.float64:
    """What a cell adds to the pull: 4 pi G times its quanta over the room it stands in, for plus energy only."""
    if not makes_pull(kind) or cell_quanta <= wp.int64(0) or not (reach > wp.float64(0.0)) or wp.isinf(reach):
        return wp.float64(0.0)
    return _FOUR_PI * pull_strength * wp.float64(cell_quanta) / (reach * reach * reach)
