"""Every scorecard scene runs on a small universe (8 lattice places a side, 64 energy volumes) and returns only finite numbers.

The exceptions are NaN by contract: black_hole's radius_energy_power when fewer than two
runs seal (at this size none does), and a q where the size did not move.
"""
import math

import pytest

from physics.scenes import black_hole, bound_lump, expansion, hubble_gap, orbit, space_dimension, structure, timescape
from physics.scenes.tests._gpu import needs_gpu

SCENES = {
    "expansion": lambda: expansion(8, steps=8, every=1),
    "hubble_gap": lambda: hubble_gap(8, steps=12, every=1),
    "structure": lambda: structure(8, steps=4, every=2),
    "bound_lump": lambda: bound_lump(8, steps=6, every=2),
    "orbit": lambda: orbit(8, distances=(2,), steps=6, every=2),
    "black_hole": lambda: black_hole(8, energies=(4.0, 8.0), steps=10),
    "space_dimension": lambda: space_dimension(8, steps=4, sources=4),
    "timescape": lambda: timescape(8, steps=12, every=2, observers=2),
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
    result.pop("radius_energy_power", None)
    numbers = list(_numbers(result))
    assert numbers
    assert all(isinstance(number, (int, float)) and math.isfinite(number) for number in numbers)
