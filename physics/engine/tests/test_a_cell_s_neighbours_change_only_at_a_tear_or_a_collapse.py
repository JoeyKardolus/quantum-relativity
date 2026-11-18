"""The grid never flickers: every cell's neighbours and every way stay as they are, step after step,
except in a step where a joint collapsed or a volume tore.

Read off `run_records`: `relations` is a digest of every neighbour entry and of the way
table, `version` counts tears and collapses.
"""
from __future__ import annotations

import pytest

from physics.engine.tests.conftest import RUNS, needs_gpu, run_records

pytestmark = needs_gpu


@pytest.mark.parametrize("run", RUNS)
def test_the_relations_change_only_in_a_step_with_an_event(run):
    records = run_records(run)
    quiet_changes = [now["step"] for before, now in zip(records, records[1:])
                     if now["relations"] != before["relations"] and now["version"] == before["version"]]
    assert quiet_changes == []


@pytest.mark.parametrize("run", RUNS)
def test_the_event_count_is_the_tears_and_collapses_and_each_left_a_record(run):
    seen = 0
    for record in run_records(run):
        limits = record["axiom_6_limits"]
        seen += len(limits["events"]) + sum(len(event.get("swallowed", [])) for event in limits["events"])
        assert record["version"] == limits["tears"] + limits["collapses"] == seen


def test_the_default_run_has_collapses_and_the_uneven_run_has_tears():
    default, uneven = run_records("default")[-1]["axiom_6_limits"], run_records("uneven")[-1]["axiom_6_limits"]
    assert default["collapses"] > 0
    assert uneven["tears"] > 0
