"""The pattern is the step's own standing swirl: a step only turns it, it winds once, its core is open."""
import numpy as np
import pytest

from physics.engine.axiom_2_swirls.definitions.standing_mode import row_step, standing_mode
from physics.engine.axiom_2_swirls.definitions.swirl_pattern import swirl_patterns
from physics.engine.axiom_2_swirls.logic.read_swirl import core_share, round_flow, whole_turns


@pytest.mark.parametrize("across", [6, 12])
def test_one_unrounded_step_turns_every_box_by_one_common_angle(across):
    mode, turn_per_step = standing_mode(across)
    layer = mode[:, None] - 1j * mode[None, :]
    stepped = row_step(across) @ layer @ row_step(across).T
    assert np.abs(stepped / layer - np.exp(1j * turn_per_step)).max() < 1e-12
    assert 0.0 < abs(turn_per_step) < 0.1


def test_each_of_the_six_patterns_winds_once_with_an_open_core():
    patterns = swirl_patterns(6)
    assert patterns[..., 0].mean() == pytest.approx(1.0)
    for laid in range(6):
        share, phase = patterns[laid, :, 0], patterns[laid, :, 1]
        assert whole_turns(phase, laid) == 2 * (laid % 2) - 1   # the phase falls the way the fluid goes
        assert round_flow(share, phase, laid) * (1 - 2 * (laid % 2)) > 0
        assert core_share(share, laid) == pytest.approx(0.141, abs=0.001)
