"""A cell's turns, layer by layer, on blocks wound a known number of turns."""
import numpy as np
import pytest

from physics.measure.winding.logic.cell_turns import cell_turns
from physics.measure.winding.logic.tests.wound_block import wound_block


@pytest.mark.parametrize("axis", [0, 1, 2])
@pytest.mark.parametrize("turns", [1, -1, 2, -2, 0])
def test_every_layer_across_the_axis_reads_the_placed_turns_and_the_other_axes_read_none(axis, turns):
    quanta, phase = wound_block(axis, turns)
    assert cell_turns(quanta, phase, axis) == [turns] * 6
    for other in {0, 1, 2} - {axis}:
        assert cell_turns(quanta, phase, other) == [0] * 6


def test_minus_quanta_read_the_same_turns_as_plus():
    quanta, phase = wound_block(1, 1)
    assert cell_turns(-quanta, phase, 1) == [1] * 6


def test_a_layer_whose_middle_box_holds_no_quanta_reads_no_turn():
    quanta, phase = wound_block(2, 1)
    quanta[2, 2, 3] = 0
    assert cell_turns(quanta, phase, 2) == [1, 1, 1, 0, 1, 1]


def test_arrays_of_unequal_shape_are_refused():
    quanta, phase = wound_block(0, 1)
    with pytest.raises(ValueError):
        cell_turns(quanta[:5], phase, 0)
    with pytest.raises(ValueError):
        cell_turns(quanta, phase, 3)
