"""A plan's long runs are sampled as many times as its short ones and given the same wall time a step; a plan without cells, steps or workers is refused."""
import pytest

from physics.fit.search.definitions.search_plan import SearchPlan


def test_the_default_plan_is_the_search_written_down():
    plan = SearchPlan()
    assert (plan.cells, plan.steps, plan.every, plan.hold_steps) == (24, 5000, 100, 20000)
    assert plan.long_every() == 400 and plan.steps // plan.every == plan.hold_steps // plan.long_every() == 50
    assert plan.long_seconds() == 960.0


def test_a_tiny_plan_and_one_without_a_wall_time_cap():
    plan = SearchPlan(cells=8, steps=8, every=2, hold_steps=16, seconds=None)
    assert plan.long_every() == 4 and plan.long_seconds() is None
    assert SearchPlan(steps=10, every=7, hold_steps=1).long_every() == 1, "never fewer than one step between samples"


def test_a_plan_with_nothing_to_run_is_refused():
    for name in ("cells", "steps", "every", "workers"):
        with pytest.raises(ValueError, match=name):
            SearchPlan(**{name: 0})
