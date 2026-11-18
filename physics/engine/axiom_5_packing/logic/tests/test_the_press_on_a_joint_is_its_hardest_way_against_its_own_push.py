"""press_joints on a real small universe: every way by the press law, each joint its hardest way, squeeze over push."""
from __future__ import annotations

import numpy as np
import warp as wp

from physics import engine
from physics.engine.axiom_1_energy.logic.clear_ledgers import clear_ledgers
from physics.engine.axiom_5_packing.logic.press_joints import press_joints

wp.config.quiet = True


def _universe(side: int = 8, **changed):
    settings = engine.Settings(**changed)
    return engine.new_universe(settings, cells_per_side=side, seed=1, device="cpu"), settings


def _far_and_reach(state, volume: int, mouth: int, joint: int) -> tuple:
    """The mean pull behind a volume, on the side away from the joint, and the reach from the joint to there."""
    pull, length, quanta = state.host("pull"), state.host("length").astype(np.float64), state.host("cell_quanta")
    behind = [one for one, code in state.neighbours(volume) if code == 26 - mouth and quanta[one] > 0]
    far = float(np.mean(pull[behind])) if behind else float(pull[volume])
    far_length = float(np.mean(length[behind])) if behind else 0.0
    return far, length[joint] / 2.0 + length[volume] + far_length / 2.0


def _press_by_hand(state, way: tuple, joint: int) -> float:
    slot, way_a, way_b, mouth_a, mouth_b = way
    quanta, pull = state.host("cell_quanta").astype(np.float64), state.host("pull")
    far_a, reach_a = _far_and_reach(state, way_a, mouth_a, joint)
    far_b, reach_b = _far_and_reach(state, way_b, mouth_b, joint)
    slopes = (far_a - pull[joint]) / reach_a + (far_b - pull[joint]) / reach_b
    return quanta[way_a] * quanta[way_b] / (quanta[way_a] + quanta[way_b]) * slopes


def _pressed(side: int = 8):
    state, settings = _universe(side)
    clear_ledgers(state)
    press_joints(state, settings)
    return state, settings


def test_every_way_s_press_is_the_press_law_worked_by_hand_from_the_one_pull():
    state, settings = _pressed()
    way_press = state.host("way_press")
    joints = np.flatnonzero(state.host("kind") == engine.JOINT)
    for joint in joints[::37].tolist():
        for way in state.ways_of(joint):
            assert np.isclose(way_press[way[0]], _press_by_hand(state, way, joint), rtol=1e-5)


def test_a_joint_s_press_is_its_hardest_way_never_a_sum_and_a_tie_goes_to_the_lowest_slot():
    state, settings = _pressed()
    way_press, press, hardest = state.host("way_press"), state.host("press"), state.host("squeezed_way")
    several = 0
    for joint in np.flatnonzero(state.host("kind") == engine.JOINT).tolist():
        slots = [way[0] for way in state.ways_of(joint)]
        largest = max(way_press[slot] for slot in slots)
        assert press[joint] == largest
        assert hardest[joint] == min(slot for slot in slots if way_press[slot] == largest)
        several += len(slots) > 1 and len({float(way_press[slot]) for slot in slots}) > 1
    assert several > 100
    volumes = state.host("kind") == engine.ENERGY_VOLUME
    assert not press[volumes].any() and (hardest[volumes] == -1).all() and not state.host("squeeze")[volumes].any()


def test_the_squeeze_is_the_press_over_the_joint_s_own_push_and_nothing_when_it_is_drawn_apart():
    state, settings = _pressed()
    press, squeeze = state.host("press").astype(np.float64), state.host("squeeze")
    quanta, length = state.host("cell_quanta").astype(np.float64), state.host("length").astype(np.float64)
    joints = state.host("kind") == engine.JOINT
    push = settings.push_strength * (quanta[joints] / length[joints]) ** 2
    assert np.allclose(squeeze[joints], np.maximum(press[joints], 0.0) / push, rtol=1e-5)
    assert (squeeze[joints][press[joints] <= 0.0] == 0.0).all() and (squeeze[joints] > 1.0).any()


def test_two_equal_volumes_on_one_even_slope_press_nothing_and_pulled_together_they_press():
    state, settings = _universe(side=8)
    joint = next(int(cell) for cell in np.flatnonzero(state.host("kind") == engine.JOINT)
                 if len(state.ways_of(int(cell))) == 1)
    (slot, way_a, way_b, mouth_a, mouth_b), = state.ways_of(joint)
    length = state.host("length").astype(np.float64)
    behind_a = next(one for one, code in state.neighbours(way_a) if code == 26 - mouth_a)
    behind_b = next(one for one, code in state.neighbours(way_b) if code == 26 - mouth_b)
    reach_a = length[joint] / 2.0 + length[way_a] + length[behind_a] / 2.0
    reach_b = length[joint] / 2.0 + length[way_b] + length[behind_b] / 2.0
    for cell in range(state.count):
        state.set_cell("pull", cell, 0.0)
    state.set_cell("pull", behind_a, 1.0 * reach_a)          # the pull falls by 1 a length from behind a ...
    state.set_cell("pull", behind_b, -1.0 * reach_b)         # ... through the joint (0) to behind b: one even slope
    clear_ledgers(state)
    press_joints(state, settings)
    scale = float(state.host("cell_quanta")[way_a])
    assert abs(state.host("way_press")[slot]) < 1e-6 * scale
    state.set_cell("pull", behind_b, 1.0 * reach_b)          # now both sides slope down to the joint
    clear_ledgers(state)
    press_joints(state, settings)
    assert state.host("way_press")[slot] > 0.1 * scale


def test_an_emptied_cell_behind_a_volume_is_empty_space_and_its_old_pull_is_not_read():
    state, settings = _universe(side=8)
    joint = next(int(cell) for cell in np.flatnonzero(state.host("kind") == engine.JOINT)
                 if len(state.ways_of(int(cell))) == 1)
    way = state.ways_of(joint)[0]
    behind = next(one for one, code in state.neighbours(way[1]) if code == 26 - way[3])
    state.set_cell("cell_quanta", behind, 0)
    state.set_cell("pull", behind, 1.0e9)
    clear_ledgers(state)
    press_joints(state, settings)
    assert abs(state.host("way_press")[way[0]]) < 1.0e6
    assert np.isclose(state.host("way_press")[way[0]], _press_by_hand(state, way, joint), rtol=1e-5)
