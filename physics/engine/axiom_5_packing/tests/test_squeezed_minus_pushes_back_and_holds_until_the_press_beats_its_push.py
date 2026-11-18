"""Squeezed minus on the real runs: a joint holds while its push beats the press, and yields only past it."""
from __future__ import annotations

from physics.engine.tests.conftest import needs_gpu, run_records

pytestmark = needs_gpu


def _largest(record: dict) -> float:
    return max(record["axiom_5_packing"]["largest_squeeze"].values())


def test_the_start_lies_in_balance_so_after_one_step_the_hardest_squeeze_is_close_to_one():
    for name in ("default", "collapsing"):
        first = _largest(run_records(name)[0])
        assert 0.9 < first < 1.1, f"{name}: largest squeeze after step 1 is {first}"


def test_a_push_too_hard_to_balance_leaves_the_joints_at_one_quantum_a_box_and_slack():
    # push 40 (the held runs): the balance would ask for less than one quantum a box, and no box starts empty
    assert 0.0 < max(_largest(record) for record in run_records("held")) < 1.0
    assert all(record["axiom_6_limits"]["made"] == 0 and record["axiom_6_limits"]["collapses"] == 0
               for record in run_records("held"))


def test_squeezed_joints_make_energy_and_hold_while_they_are_below_their_limit():
    records = run_records("default")
    assert any(record["axiom_6_limits"]["made"] > 0 for record in records)
    assert all(sum(record["axiom_5_packing"]["squeezed_joints"].values()) > 0
               for record in records if record["axiom_6_limits"]["made"] > 0), "only a squeezed joint makes energy"
    assert all(record["axiom_6_limits"]["collapses"] == 0 for record in records), "none is squeezed to its limit of 3"
    assert max(_largest(record) for record in records) < 1.5


def test_with_a_limit_a_hundredth_above_fresh_the_joints_that_made_energy_collapse():
    records = run_records("collapsing")
    first = next(index for index, record in enumerate(records) if record["axiom_6_limits"]["collapses"] > 0)
    assert first >= 1 and any(record["axiom_6_limits"]["made"] > 0 for record in records[:first])
