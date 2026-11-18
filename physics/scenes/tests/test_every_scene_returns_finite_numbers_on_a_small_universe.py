"""Every scorecard scene runs on a small standing universe and returns only finite numbers.

The exceptions are NaN by contract: a power from fewer than two readings (black hole,
two lumps) and a q where the fitted size did not curve.
"""
import math

import pytest

from physics.scenes import SCENES
from physics.scenes.tests._gpu import HELD, needs_gpu

SMALL = {
    "expansion": {"cells_per_side": 8, "steps": 4, "every": 1},
    "hubble_gap": {"cells_per_side": 8, "steps": 4, "every": 1},
    "structure": {"cells_per_side": 8, "steps": 2, "every": 1},
    "bound_lump": {"cells_per_side": 8, "steps": 2, "every": 1},
    "orbit": {"cells_per_side": 16, "distances": (2,), "steps": 2, "every": 1},
    "black_hole": {"cells_per_side": 8, "energies": (4.0,), "steps": 2, "window": 2, "every": 1},
    "dimension": {"cells_per_side": 16, "steps": 1, "sources": 4},
    "timescape": {"cells_per_side": 8, "steps": 4, "every": 1, "observers": 2},
    "isotropy": {"cells_per_side": 16, "steps": 2, "settle": 1, "pairs": (((3, 0, 0), (2, 2, 1)),),
                 "diagonals": ((2, 0, 0), (2, 2, 2))},
    "spin": {"cells_per_side": 8, "steps": 2, "count": 1},
    "quantum_split": {"cells_per_side": 8, "wait": 1},
    "two_lumps": {"cells_per_side": 12, "apart": (2,), "steps": 2, "settle": 1},
    "gravity": {"cells_per_side": 24, "steps": 2, "every": 1},
}
NAN_BY_CONTRACT = ("held_power", "depth_power", "depth_law_power", "pull_power", "newton_power")


def _numbers(value):
    """Every plain number in a nested result."""
    if isinstance(value, dict):
        for inner in value.values():
            yield from _numbers(inner)
    elif isinstance(value, (list, tuple)):
        for inner in value:
            yield from _numbers(inner)
    else:
        yield value


def test_the_scorecard_has_its_thirteen_scenes_and_each_has_a_small_run():
    assert sorted(SCENES) == sorted(SMALL) and len(SCENES) == 13


@needs_gpu
@pytest.mark.parametrize("name", sorted(SMALL))
def test_the_scene_returns_only_finite_numbers(name):
    result = SCENES[name](settings=HELD, **SMALL[name])
    for power in NAN_BY_CONTRACT:
        result.pop(power, None)
    numbers = list(_numbers(result))
    assert numbers
    odd = [number for number in numbers if not (isinstance(number, (int, float)) and math.isfinite(number))]
    assert not odd, f"{name} returned {odd[:5]}"
