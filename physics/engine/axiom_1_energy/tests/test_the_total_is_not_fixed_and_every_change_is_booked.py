"""The total changes only by what joints make and collapses take, each count kept whole and never run backwards.

The books themselves (`book_amount`, `book_plus`, `book_minus`, `book_taken`) are
asserted on every step of every run in test_axiom_1_holds_at_every_step_of_the_real_runs.py;
here is how the counts behind them move from step to step.
"""
from __future__ import annotations

import pytest

from physics.engine.tests.conftest import RUNS, needs_gpu, run_records

pytestmark = needs_gpu

EVERY_RUN = pytest.mark.parametrize("name", sorted(RUNS))


def _of(name: str) -> list:
    return [record["axiom_1_energy"] for record in run_records(name)]


@EVERY_RUN
def test_what_joints_made_each_step_adds_up_to_the_total_made(name):
    made_so_far = 0
    for step, record in enumerate(_of(name), start=1):
        made_so_far += record["made"]
        assert record["made"] >= 0 and record["made_total"] == made_so_far, f"step {step} of the {name} run"


@EVERY_RUN
def test_the_counts_that_change_the_total_never_run_backwards(name):
    records = _of(name)
    for key in ("made_total", "taken_joint", "taken_volumes", "short_takes", "short_quanta", "torn"):
        values = [record[key] for record in records]
        assert all(later >= earlier for earlier, later in zip(values, values[1:])), key


@EVERY_RUN
def test_minus_only_grows_by_making_and_tearing_and_plus_never_grows(name):
    records = _of(name)
    for earlier, later in zip(records, records[1:]):
        assert later["plus"] <= earlier["plus"]
        assert later["minus"] - earlier["minus"] <= later["made"] + (later["torn"] - earlier["torn"])


@EVERY_RUN
def test_a_collapse_takes_off_the_rest_what_its_joint_held_less_only_what_was_not_there(name):
    last = _of(name)[-1]
    assert last["taken_volumes"] == last["taken_joint"] - last["short_quanta"]
    assert last["short_quanta"] == 0 or last["short_takes"] > 0
