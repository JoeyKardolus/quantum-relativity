"""The turns a whole swirl reads, as a table; the tie to the engine itself is tested in physics/measure/tests."""
import pytest

from physics.measure.winding.definitions.expected_turns import expected_turns


@pytest.mark.parametrize("axis, sense, turns", [(0, 1, -1), (0, -1, 1), (1, 1, 1), (1, -1, -1), (2, 1, -1), (2, -1, 1)])
def test_the_turns_are_minus_the_sense_except_about_axis_one(axis, sense, turns):
    assert expected_turns(axis, sense) == turns


@pytest.mark.parametrize("axis, sense", [(3, 1), (0, 0), (0, 2)])
def test_another_axis_or_sense_is_refused(axis, sense):
    with pytest.raises(ValueError):
        expected_turns(axis, sense)
