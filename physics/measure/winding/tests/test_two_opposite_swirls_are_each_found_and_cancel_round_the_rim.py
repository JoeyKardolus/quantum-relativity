"""The whole measure on a field with two swirls of opposite turns: each is found whole in its own
square, and a layer's total, the turns round its rim, is 0."""
import numpy as np

from physics.measure.winding import cell_turns, phase_winding

SIDE = 24
PLUS, MINUS = (6.5, 6.5), (16.5, 15.5)


def _pair():
    first, second = np.meshgrid(np.arange(SIDE, dtype=np.float64), np.arange(SIDE, dtype=np.float64), indexing="ij")
    angle = np.arctan2(second - PLUS[1], first - PLUS[0]) - np.arctan2(second - MINUS[1], first - MINUS[0])
    return np.full((SIDE,) * 3, 9, dtype=np.int32), np.repeat(np.angle(np.exp(1j * angle))[:, :, None], SIDE, axis=2)


def test_the_plus_and_the_minus_swirl_sit_in_their_own_squares_and_the_layers_total_none():
    quanta, phase = _pair()
    windings = phase_winding(np.sqrt(quanta) * np.exp(1j * phase))
    assert not windings[0].any() and not windings[1].any()
    assert set(map(tuple, np.argwhere(windings[2] == 1)[:, :2].tolist())) == {(6, 6)}
    assert set(map(tuple, np.argwhere(windings[2] == -1)[:, :2].tolist())) == {(16, 15)}
    assert np.count_nonzero(windings[2]) == 2 * SIDE
    assert cell_turns(quanta, phase, 2) == [0] * SIDE


def test_with_the_minus_swirl_s_boxes_emptied_only_the_plus_one_is_counted():
    quanta, phase = _pair()
    quanta[16:18, 15:17, :] = 0
    assert cell_turns(quanta, phase, 2) == [1] * SIDE
