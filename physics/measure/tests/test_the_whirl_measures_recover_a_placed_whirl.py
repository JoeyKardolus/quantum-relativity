"""On a field built with one whirl of known turns and direction, the winding is that whole number,
the line points along the whirl, and the flow circulation and strength come within 10% of it."""
import numpy as np
import pytest

from physics.measure import loop_circulation, phase_winding, whirl_line, whirl_strength

SIDE = 24
CENTRE = np.array([11.5, 11.3, 11.6])


def _whirl(direction, turns):
    """A whirl of `turns` whole turns along `direction` through CENTRE, density 0 on its line."""
    grid = np.stack(np.meshgrid(*[np.arange(SIDE, dtype=np.float64)] * 3, indexing="ij"), axis=-1) - CENTRE
    direction = np.asarray(direction, dtype=np.float64) / np.linalg.norm(direction)
    across = grid - (grid @ direction)[..., None] * direction
    first = np.cross(direction, [1.0, 0.0, 0.0] if abs(direction[0]) < 0.9 else [0.0, 1.0, 0.0])
    first /= np.linalg.norm(first)
    second = np.cross(direction, first)
    angle = np.arctan2(across @ second, across @ first)
    return np.tanh(np.linalg.norm(across, axis=-1) / 1.5) * np.exp(1j * turns * angle)


@pytest.mark.parametrize("direction, turns", [((0, 0, 1), 1), ((0, 0, 1), 2), ((1, 0, 0), -1),
                                              ((0.3, 0.2, 1.0), 1)])
def test_a_placed_whirl_is_found_whole_and_pointing_its_way(direction, turns):
    field = _whirl(direction, turns)
    unit = np.asarray(direction, dtype=np.float64) / np.linalg.norm(direction)
    line = whirl_line(phase_winding(field), near=CENTRE, reach=8.0)
    assert line["found"] == 1
    assert np.degrees(np.arccos(np.clip(line["axis"] @ unit * np.sign(turns), -1.0, 1.0))) <= 5.0
    assert np.linalg.norm(line["centre"] - CENTRE) <= 1.0
    loop = loop_circulation(field, CENTRE, int(np.argmax(np.abs(unit))), 4)
    assert loop["winding"] == turns
    assert abs(loop["flow_circulation"] - turns) <= 0.1 * abs(turns)
    assert abs(whirl_strength(field, CENTRE, unit, 6.0, 3.0) - turns) <= 0.1 * abs(turns)


def test_an_even_phase_has_no_winding_anywhere():
    field = np.exp(1j * 0.3 * np.arange(SIDE))[:, None, None] * np.ones((SIDE, SIDE, SIDE))
    assert not phase_winding(field).any()
    assert whirl_line(phase_winding(field))["found"] == 0
