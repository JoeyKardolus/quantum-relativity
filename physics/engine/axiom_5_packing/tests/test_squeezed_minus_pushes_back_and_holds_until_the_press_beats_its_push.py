"""Squeezed minus on the real runs: a joint holds while its push beats the press, and yields only past it."""
from __future__ import annotations

from physics.engine.tests.conftest import needs_gpu, run_records

pytestmark = needs_gpu


def _largest(record: dict) -> float:
    return max(record["axiom_5_packing"]["largest_squeeze"].values())


def test_with_a_push_that_beats_the_press_no_joint_is_squeezed_and_none_makes_energy_or_collapses():
    records = run_records("held")
    assert 0.0 < max(_largest(record) for record in records) < 1.0
    assert all(sum(record["axiom_5_packing"]["squeezed_joints"].values()) == 0 for record in records)
    assert all(record["axiom_6_limits"]["made"] == 0 and record["axiom_6_limits"]["collapses"] == 0
               for record in records)


def test_at_the_default_push_the_press_beats_it_and_the_squeezed_joints_make_energy_at_once():
    first = run_records("default")[0]
    assert first["axiom_6_limits"]["made"] > 0 and first["axiom_6_limits"]["collapses"] > 0


def test_once_the_joints_that_made_energy_have_collapsed_the_largest_squeeze_is_lower_than_at_the_start():
    records = run_records("default")
    making = [record for record in records if record["axiom_6_limits"]["made"] > 0]
    assert len(making) >= 2
    assert _largest(records[len(making)]) < _largest(making[0])
