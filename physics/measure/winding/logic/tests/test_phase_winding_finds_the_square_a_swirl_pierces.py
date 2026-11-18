"""The unit squares: one whole turn sits in the one square the swirl's line goes through."""
import numpy as np
import pytest

from physics.measure.winding.logic.phase_winding import phase_winding
from physics.measure.winding.logic.tests.wound_block import wound_block


def _field(axis, turns):
    quanta, phase = wound_block(axis, turns)
    return np.sqrt(quanta) * np.exp(1j * phase)


@pytest.mark.parametrize("axis", [0, 1, 2])
@pytest.mark.parametrize("turns", [1, -1])
def test_one_turn_is_in_the_square_round_the_line_in_every_layer_and_nowhere_else(axis, turns):
    windings = phase_winding(_field(axis, turns))
    pierced = np.argwhere(windings != 0)
    assert len(pierced) == 6
    assert set(windings[tuple(pierced.T)].tolist()) == {turns}
    assert set(pierced[:, 0].tolist()) == {axis}
    square = np.delete(pierced[:, 1:], axis, axis=1)
    assert set(map(tuple, square.tolist())) == {(2, 2)}


def test_an_even_phase_and_a_phase_that_only_ramps_wind_nowhere():
    ramp = np.exp(1j * 0.3 * np.arange(6))[:, None, None] * np.ones((6, 6, 6))
    assert not phase_winding(np.ones((6, 6, 6), dtype=complex)).any()
    assert not phase_winding(ramp).any()


def test_a_square_touching_a_box_without_phase_or_outside_the_mask_holds_nothing():
    field = _field(2, 1)
    emptied = field.copy()
    emptied[2, 2, 0] = 0.0
    assert phase_winding(emptied)[2, :, :, 0].sum() == 0 and phase_winding(emptied)[2, :, :, 1].sum() == 1
    valid = np.ones((6, 6, 6), dtype=bool)
    valid[3, 3, 5] = False
    assert phase_winding(field, valid)[2, :, :, 5].sum() == 0 and phase_winding(field, valid)[2, :, :, 4].sum() == 1


def test_a_field_that_is_not_three_dimensional_is_refused():
    with pytest.raises(ValueError):
        phase_winding(np.ones((6, 6), dtype=complex))
