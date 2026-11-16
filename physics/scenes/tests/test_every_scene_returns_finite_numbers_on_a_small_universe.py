"""Every scorecard scene runs on a small universe (4 to 7 cells of 4 boxes a side) and returns only finite numbers.

The one exception is black_hole's radius_energy_power, NaN by contract when fewer than two
runs seal (at this size none does).
"""
import math

import pytest

from physics.scenes import (belt_trick, black_hole, bound_lump, expansion, hubble_gap, isotropy, orbit,
                            space_dimension, spin, structure)
from physics.scenes.tests._gpu import needs_gpu

SCENES = {
    "expansion": lambda: expansion(4, 4, steps=8, every=1),
    "hubble_gap": lambda: hubble_gap(4, 4, steps=8, every=1),
    "structure": lambda: structure(4, 4, steps=4, every=2),
    "bound_lump": lambda: bound_lump(4, 4, steps=6, every=2),
    "orbit": lambda: orbit(7, 4, distances=(8.0,), big_energy=100.0, big_radius=2.0, small_energy=2.0,
                           small_radius=1.5, steps=6),
    "black_hole": lambda: black_hole(6, 4, energies=(100.0, 400.0), radius=3.0, steps=20),
    "space_dimension": lambda: space_dimension(6, 4, steps=4, sources=4),
    "isotropy": lambda: isotropy(6, 4, blob_energy=100.0, blob_radius=2.0, flow_radius=4.0, speed=0.2, steps=4,
                                  seeds=(1, 2)),
    "spin": lambda: spin(4, 4, drift_speed=0.2, steps=6),
    "belt_trick": lambda: belt_trick(5, 4, margin=6, steps_per_turn=3, web_reach=2),
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
