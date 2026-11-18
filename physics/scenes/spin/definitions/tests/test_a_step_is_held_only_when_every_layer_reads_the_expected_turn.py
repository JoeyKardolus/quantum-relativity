"""Three steps of readings with one layer off at step 3 are held for 2 steps; the watched cells are diagonal volumes and the joint beside each."""
import pytest

from physics.scenes.spin.definitions.pass_mark import NOT_ASKED, passes
from physics.scenes.spin.definitions.spin_cells import spin_cells
from physics.scenes.spin.definitions.turn_record import turn_record


def test_one_layer_off_makes_the_step_not_held():
    record = turn_record([[1] * 6, [1] * 6, [1, 1, 0, 1, 1, 1], [1] * 6], 1)
    assert record == {"seen": [0, 1], "held_steps": 3, "first_off": 3}
    assert turn_record([[-1] * 6] * 4, -1) == {"seen": [-1], "held_steps": 4, "first_off": -1}
    assert turn_record([[1] * 6] * 2, -1) == {"seen": [1], "held_steps": 0, "first_off": 1}
    assert turn_record([], 1) == {"seen": [], "held_steps": 0, "first_off": -1}


def test_the_watched_cells_are_volumes_four_places_apart_and_the_joint_after_each():
    assert spin_cells(8, 2) == ([0, 292], [1, 293])
    assert spin_cells(16, 1) == ([0], [1])
    with pytest.raises(ValueError):
        spin_cells(8, 3)
    with pytest.raises(ValueError):
        spin_cells(8, 0)


def test_the_mark_wants_every_cell_whole_at_birth_and_at_every_step():
    held = {"born": [1] * 6, "expected": 1, "held_steps": 10}
    assert passes({"steps": 10, "cases": [held, {**held, "born": [-1] * 6, "expected": -1}]})
    assert not passes({"steps": 10, "cases": [held, {**held, "held_steps": 9}]})
    assert not passes({"steps": 10, "cases": [{**held, "born": [1, 1, 1, 0, 1, 1]}]})
    assert not passes({"steps": 10, "cases": []}) and len(NOT_ASKED) == 1
