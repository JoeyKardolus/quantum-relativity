"""A task's id is phase:index:seed, its overrides are its point's settings, the defaults have none, and refine rounds narrow by 0.7."""
import numpy as np
import pytest

from physics.fit.search.definitions.make_task import make_task
from physics.fit.search.definitions.refine_width import refine_width
from physics.fit.search.definitions.searched_phase import searched_phase
from physics.fit.search.definitions.unit_to_overrides import unit_to_overrides


def test_a_point_task_and_a_defaults_task():
    task = make_task("refine2", 5, np.full(12, 0.5), 3)
    assert task["id"] == "refine2:5:3" and task["phase"] == "refine2" and task["seed"] == 3
    assert task["unit"] == [0.5] * 12 and all(type(value) is float for value in task["unit"])
    assert task["overrides"] == unit_to_overrides([0.5] * 12)
    defaults = make_task("defaults", 0, None, 4)
    assert defaults == {"id": "defaults:0:4", "phase": "defaults", "seed": 4, "unit": None, "overrides": {}}


def test_refine_rounds_start_015_wide_and_narrow_by_07_each():
    assert [refine_width(round_index) for round_index in (1, 2, 3)] == pytest.approx([0.15, 0.105, 0.0735])
    with pytest.raises(ValueError):
        refine_width(0)


def test_only_the_random_and_refine_phases_are_searched_points():
    assert [searched_phase(phase) for phase in ("random", "refine1", "refine3", "defaults", "hold", "sweep")] == \
        [True, True, True, False, False, False]
