"""The ball whirl is the step's own standing wave: a step only turns it; it winds once about each direction."""
import numpy as np
import pytest

from physics.engine.axiom_2_swirls.definitions.standing_mode import row_step, standing_mode
from physics.engine.axiom_2_swirls.definitions.swirl_pattern import swirl_patterns
from physics.engine.axiom_2_swirls.logic.read_swirl import core_share, round_flow, whole_turns


@pytest.mark.parametrize("across", [6, 12])
def test_one_unrounded_step_turns_every_box_by_one_common_angle(across):
    mode, turn_per_step = standing_mode(across)
    row, third = row_step(across), np.exp(2j * np.pi / 3)

    def whirl(first, second, last):
        pairs = (np.einsum("i,j,k->ijk", first, second, np.ones(across)),
                 np.einsum("i,j,k->ijk", np.ones(across), second, last),
                 np.einsum("i,j,k->ijk", first, np.ones(across), last))
        return pairs[0] + third * pairs[1] + third ** 2 * pairs[2]
    stepped = np.einsum("ia,jb,kc,abc->ijk", row, row, row, whirl(mode, mode, mode))
    assert np.abs(stepped - np.exp(2j * turn_per_step) * whirl(mode, mode, mode)).max() < 1e-12
    assert 0.0 < abs(turn_per_step) < 0.1
    shares = swirl_patterns(across)[0, :, 0]
    assert np.allclose(shares * np.mean(np.abs(stepped) ** 2), np.abs(stepped.ravel()) ** 2)


def test_the_whirl_winds_once_about_each_direction_and_is_the_same_along_all_three():
    patterns = swirl_patterns(6)
    assert patterns[..., 0].mean() == pytest.approx(1.0)
    share, phase = patterns[0, :, 0].reshape(6, 6, 6), patterns[0, :, 1]
    assert [whole_turns(phase, axis) for axis in range(3)] == [-1, -1, -1]  # the phase falls
    assert all(round_flow(share.ravel(), phase, axis) > 0 for axis in range(3))
    assert np.allclose(share, share.transpose(1, 2, 0)) and np.allclose(share, share[::-1, ::-1, ::-1])
    for axis in range(3):  # dark on the line along each direction, and seen along it round it
        assert core_share(share.ravel(), axis) == pytest.approx(0.101, abs=0.001)
        assert share.sum(axis=axis)[2:4, 2:4].max() < 0.2 * share.sum(axis=axis).mean()
    for laid in (1, 2, 3):  # half round about a direction: the same turn about it, the others back
        turns = [whole_turns(patterns[laid, :, 1], axis) for axis in range(3)]
        assert turns == [-1 if axis == laid - 1 else 1 for axis in range(3)]
        assert np.allclose(patterns[laid, :, 0].reshape(6, 6, 6), np.flip(share, axis=laid - 1))
