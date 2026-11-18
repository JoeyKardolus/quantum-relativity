"""Axiom 6 on the real runs: a joint's energy only rises, only while squeezed; past its limit it is gone; plus tears."""
from __future__ import annotations

import pytest

from physics.engine.tests.conftest import RUNS, first_bad, needs_gpu, run_records

pytestmark = needs_gpu

EVERY_RUN = pytest.mark.parametrize("name", sorted(RUNS))


def _of(name: str) -> list:
    return [record["axiom_6_limits"] for record in run_records(name)]


def _events(name: str, what: str) -> list:
    return [event for record in _of(name) for event in record["events"] if event["what"] == what]


@EVERY_RUN
@pytest.mark.parametrize("key", ["making_unsqueezed", "joints_below_fresh", "joints_fell", "joints_rose_unmade",
                                 "joints_past_limit"])
def test_a_joint_only_rises_only_while_squeezed_and_none_stands_past_its_limit(name, key):
    counts = [record[key] for record in _of(name)]
    assert not any(counts), first_bad(counts)


@EVERY_RUN
def test_the_rise_of_every_joint_is_compared_from_the_very_first_step(name):
    assert _of(name)[0]["joint_steps_compared"] > 0


def test_in_the_default_run_joints_do_make_energy_and_that_step_is_compared():
    making = [record for record in _of("default") if record["made"] > 0]
    assert making and all(record["joint_steps_compared"] > 0 for record in making)


def test_real_collapses_take_the_joint_s_energy_off_the_rest_and_merge_two_volumes():
    collapses = _events("collapsing", "collapse")
    assert collapses, "the collapsing run had no collapse"
    merging = [event for event in collapses if event["kept"] != event["gone"]]
    assert merging and all(event["kept"] < event["gone"] for event in merging)
    assert all(event["taken_volume"] <= event["taken_joint"] for event in collapses)
    last = _of("collapsing")[-1]
    books = run_records("collapsing")[-1]["books"]
    assert books["taken_joint"] == sum(event["taken_joint"] for event in collapses)
    assert books["taken_volumes"] == sum(event["taken_volume"] for event in collapses)
    assert last["collapses"] == len(collapses) and last["merges"] == len(merging)


def test_every_collapse_was_of_a_joint_squeezed_to_its_limit_and_no_other_joint_went_with_it():
    records = _of("collapsing")
    gone = sum(len([event for event in record["events"] if event["what"] == "collapse"]) for record in records)
    joints_at_start = 16 ** 3 - 8 ** 3
    joints_left = run_records("collapsing")[-1]["axiom_3_structure"]["joints"]
    assert joints_at_start - joints_left == gone


def test_a_real_tear_splits_only_that_volume_and_its_middle_becomes_the_new_joint():
    tears = _events("uneven", "tear")
    assert tears, "the uneven run had no tear"
    for event in tears:
        assert len({event["volume"], event["joint"], event["new_volume"]}) == 3 and min(event["quanta"]) > 0
        assert event["joint"] > event["volume"] and event["new_volume"] > event["volume"]
        assert event["axis"] in (0, 1, 2)
    assert run_records("uneven")[-1]["books"]["torn"] == sum(event["quanta"][1] for event in tears)
    assert _of("uneven")[-1]["tears"] == len(tears)
