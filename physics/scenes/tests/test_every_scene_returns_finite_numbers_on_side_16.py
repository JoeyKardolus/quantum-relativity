"""Every scorecard scene runs on a side-16 grid and returns only finite numbers."""
import math

import pytest

from physics.scenes import black_hole, bound_lump, expansion, hubble_gap, orbit, structure
from physics.scenes.tests._gpu import needs_gpu

SCENES = {
    "expansion": lambda: expansion(side=16, steps=8, every=1),
    "hubble_gap": lambda: hubble_gap(side=16, steps=8, every=1),
    "structure": lambda: structure(side=16, steps=4, every=2),
    "bound_lump": lambda: bound_lump(side=16, steps=6, every=2),
    "orbit": lambda: orbit(side=16, distances=(5.0, 6.0), big_energy=100.0, big_radius=1.0, small_energy=2.0,
                           small_radius=1.0, steps=6),
    "black_hole": lambda: black_hole(side=16, energies=(100.0, 400.0), radius=3.0, steps=20),
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
    numbers = list(_numbers(SCENES[name]()))
    assert numbers
    assert all(isinstance(number, (int, float)) and math.isfinite(number) for number in numbers)
