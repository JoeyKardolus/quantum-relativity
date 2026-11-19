"""The ring whirl is the step's own standing wave: a step only turns it; one whole turn in each ring, all alike."""
import numpy as np
import pytest

from physics.engine.axiom_2_swirls.definitions.ring_boxes import cap_boxes
from physics.engine.axiom_2_swirls.definitions.standing_mode import ring_step, standing_mode
from physics.engine.axiom_2_swirls.definitions.swirl_pattern import swirl_patterns
from physics.engine.axiom_2_swirls.logic.read_swirl import round_flow, whole_turns


@pytest.mark.parametrize("across", [6, 12, 24])
def test_one_unrounded_step_turns_every_box_of_the_ring_by_one_common_angle(across):
    mode, turn_per_step = standing_mode(across)
    stepped = ring_step(across) @ mode
    assert np.abs(stepped - np.exp(1j * turn_per_step) * mode).max() < 1e-12
    assert np.allclose(np.abs(mode), 1.0)  # every box holds the same
    falls = np.angle(mode / np.roll(mode, -1))
    assert (falls > 0).all() and falls.sum() == pytest.approx(2 * np.pi)  # one whole turn, falling
    mirror = mode[::-1]  # the other way round: a standing wave of the same step, the same turn
    assert np.abs(ring_step(across) @ mirror - np.exp(1j * turn_per_step) * mirror).max() < 1e-12
    assert turn_per_step * across ** 2 == pytest.approx(9.5, abs=0.6)  # a turn: across squared steps


def test_a_turn_is_25_steps_at_6_boxes_across():
    assert 2 * np.pi / standing_mode(6)[1] == pytest.approx(24.94, abs=0.01)


def test_the_whirl_winds_once_in_each_ring_the_three_alike_and_a_joint_s_is_its_mirror():
    patterns = swirl_patterns(6)
    assert np.allclose(patterns[..., 0], 1.0)  # every box an even share: no empty box, no empty line
    share, phase = patterns[0, :, 0], patterns[0, :, 1]
    assert [whole_turns(phase, ring) for ring in range(3)] == [-1, -1, -1]  # the phase falls
    flows = [round_flow(share, phase, ring) for ring in range(3)]
    assert flows[0] > 0 and flows[0] == flows[1] == flows[2]
    rings = phase.reshape(3, 72)
    assert np.array_equal(rings[0], rings[1]) and np.array_equal(rings[0], rings[2])  # box for box
    minus = patterns[1, :, 1]
    assert [whole_turns(minus, ring) for ring in range(3)] == [1, 1, 1]
    assert np.array_equal(minus.reshape(6, 6, 6), phase.reshape(6, 6, 6)[:, ::-1, :])


@pytest.mark.parametrize("across", [6, 12])
def test_a_joint_s_caps_hold_the_very_phases_of_the_volumes_caps_they_face(across):
    volume, joint = swirl_patterns(across)[:, :, 1]
    for axis in range(3):
        front, back = cap_boxes(across, axis, 1), cap_boxes(across, axis, 0)
        assert len(front) == len(set(front)) == across * (across // 6) and not set(front) & set(back)
        assert np.array_equal(volume[front], joint[back]) and np.array_equal(joint[front], volume[back])
        apart = np.exp(1j * (volume[front] - volume[back])).mean()  # two bare volumes: half a turn
        assert abs(np.angle(-apart)) < 0.02 and apart.real < -0.85  # (-0.9999 at 6 across, -0.859 at 12)
