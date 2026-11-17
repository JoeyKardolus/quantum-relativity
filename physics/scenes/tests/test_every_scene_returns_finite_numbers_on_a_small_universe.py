"""Every scorecard scene runs on a small universe (8 lattice places a side, 64 energy volumes) and returns only finite numbers.

The exceptions are NaN by contract: black_hole's three powers when fewer than two runs
give a radius above 0 and under the box's limit, a q where the size did not move, and
two_lumps' powers from a single distance.
"""
import math

import pytest

from physics.scenes import (black_hole, bound_lump, expansion, hubble_gap, isotropy, orbit, quantum_split, space_dimension,
                            spin, structure, timescape, two_lumps)
from physics.scenes.tests._gpu import needs_gpu

SCENES = {
    "expansion": lambda: expansion(8, steps=8, every=1),
    "hubble_gap": lambda: hubble_gap(8, steps=12, every=1),
    "structure": lambda: structure(8, steps=4, every=2),
    "bound_lump": lambda: bound_lump(8, steps=6, every=2),
    "orbit": lambda: orbit(8, distances=(2,), steps=6, every=2),
    "black_hole": lambda: black_hole(8, energies=(4.0, 8.0), steps=10, window=4, every=2),
    "space_dimension": lambda: space_dimension(8, steps=4, sources=4),
    "timescape": lambda: timescape(8, steps=12, every=2, observers=2),
    "isotropy": lambda: isotropy(16, steps=6, settle=2, pairs=(((3, 0, 0), (2, 2, 1)),), diagonals=((2, 0, 0), (2, 2, 2))),
    "spin": lambda: spin(8, steps=6, cases=(("volume", 1), ("joint", -1))),
    "quantum_split": lambda: quantum_split(8, draws=8),
    "two_lumps": lambda: two_lumps(12, apart=(2,), steps=6, settle=2),
}


def _numbers(value):
    if isinstance(value, dict):
        for inner in value.values():
            yield from _numbers(inner)
    elif isinstance(value, (list, tuple)):
        for inner in value:
            yield from _numbers(inner)
    else:
        yield value


@needs_gpu
@pytest.mark.parametrize("name", sorted(SCENES))
def test_the_scene_returns_only_finite_numbers(name):
    result = SCENES[name]()
    for power in ("held_power", "depth_power", "depth_law_power", "pull_power", "newton_power"):
        result.pop(power, None)
    numbers = list(_numbers(result))
    assert numbers
    assert all(isinstance(number, (int, float)) and math.isfinite(number) for number in numbers)
