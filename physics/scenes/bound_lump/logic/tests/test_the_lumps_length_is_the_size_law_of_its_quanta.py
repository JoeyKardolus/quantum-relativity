"""The fullest volume's length at the end is the size constant over its quanta, and it is shorter than the others."""
import pytest

from physics.scenes.bound_lump.logic.bound_lump import bound_lump
from physics.scenes.tests._gpu import HELD, needs_gpu


@needs_gpu
def test_the_fullest_volume_is_as_long_as_its_energy_says_and_shorter_than_the_rest():
    result = bound_lump(8, steps=4, every=2, settings=HELD)
    quanta = result["lump_quanta_factor"] * 648000.0
    assert result["lump_length"][-1] == pytest.approx(HELD.size_constant / quanta, rel=1e-5)
    assert result["lump_length"][-1] < result["rest_length"][-1], "more energy, smaller"
    assert result["lump_quanta_factor"] > 1.0 and len(result["lump_length"]) == 3
    assert result["lump_growth"] == pytest.approx(result["lump_length"][-1] / result["lump_length"][0])
    assert result["rest_growth"] == pytest.approx(result["rest_length"][-1] / result["rest_length"][0])
