"""Axiom 2 on the real runs: every cell's own fluid goes round its axis its own way; plus only pulls, minus is not pulled."""
from __future__ import annotations

import pytest

from physics.engine.tests.conftest import RUNS, first_bad, needs_gpu, run_records

pytestmark = needs_gpu

EVERY_RUN = pytest.mark.parametrize("name", sorted(RUNS))

#: A clause the engine does not keep yet (this axiom's CLAUDE.md, "Not done": no rule makes or keeps a cell a swirl).
#: Strict: the day it holds, the mark turns the test red and must go.
NOT_KEPT = pytest.mark.xfail(strict=True, reason="not kept: nothing keeps a cell a swirl through a collapse, a tear "
                                               "or a fill of some 20 quanta a box (axiom 2, Not done)")
#: Since 2026-11-18 the start's swirls mesh and stand: the volumes keep theirs in every run that stands. Lost:
#: the volumes of the runs with events, and the joints where the balance leaves them few quanta (push 40).
_LOST = {("volumes", "collapsing"), ("volumes", "uneven"), ("joints", "uneven"), ("joints", "held"), ("joints", "gentle")}


def _cases(count: str) -> list:
    """Every run for the volumes' and the joints' count, the cases the engine does not keep marked."""
    return [pytest.param(f"{count}_{kind}", name, marks=[NOT_KEPT] if (kind, name) in _LOST else [])
            for kind in ("volumes", "joints") for name in sorted(RUNS)]


def _of(name: str) -> list:
    return [record["axiom_2_swirls"] for record in run_records(name)]


@pytest.mark.parametrize("key,name", _cases("against"))
def test_every_cell_s_own_fluid_goes_round_its_axis_the_way_its_sense_says(key, name):
    counts = [record[key] for record in _of(name)]
    assert not any(counts), first_bad(counts)


@pytest.mark.parametrize("key,name", _cases("unwound"))
def test_the_phase_keeps_going_exactly_once_round_the_axis_a_whole_turn(key, name):
    counts = [record[key] for record in _of(name)]
    assert not any(counts), first_bad(counts)


@EVERY_RUN
@pytest.mark.parametrize("key", ["joint_pull_made", "joint_pull_felt"])
def test_no_joint_makes_the_pull_and_no_joint_s_fluid_is_turned_by_it(name, key):
    counts = [record[key] for record in _of(name)]
    assert not any(counts), first_bad(counts)
