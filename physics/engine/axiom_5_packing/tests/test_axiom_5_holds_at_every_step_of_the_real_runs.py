"""Axiom 5 on the real runs: size, packing and clock follow the energy; the more packed sit deeper and energy moves to them."""
from __future__ import annotations

import math

import pytest

from physics.engine.tests.conftest import RUNS, first_bad, needs_gpu, run_records

pytestmark = needs_gpu

EVERY_RUN = pytest.mark.parametrize("name", sorted(RUNS))


def _not_kept(why: str):
    """Marks a clause the engine does not keep yet (this axiom's CLAUDE.md, "Not done"); strict, so the day it holds the mark must go."""
    return pytest.mark.xfail(strict=True, reason=f"not kept: {why} (axiom 5, Not done)")


def _of(name: str) -> list:
    return [record["axiom_5_packing"] for record in run_records(name)]


@EVERY_RUN
@pytest.mark.parametrize("key", ["stale_cells", "pull_not_finite"])
def test_no_cell_keeps_a_size_packing_or_clock_from_before_its_energy_changed(name, key):
    counts = [record[key] for record in _of(name)]
    assert not any(counts), first_bad(counts)


@EVERY_RUN
def test_the_more_packed_volumes_sit_deeper_in_the_pull_from_the_first_step(name):
    assert _of(name)[0]["pull_follows_packing"] < 0.0


def test_the_more_packed_volumes_stay_deeper_while_the_grid_stands():
    follows = [record["pull_follows_packing"] for record in _of("held")]
    assert max(follows) < 0.0, f"largest {max(follows)} at step {follows.index(max(follows)) + 1}"


@_not_kept("energy gain does not follow depth at step 1 of the held run")
def test_on_a_grid_that_stands_energy_moves_toward_the_deeper_pull_in_every_step():
    records = _of("held")
    goes = [record["gain_goes_with_depth"] for record in records]
    assert min(goes) > 0.0, f"least {min(goes)} at step {goes.index(min(goes)) + 1}"
    assert all(record["fuller_gain"] > 0 for record in records), "the fuller half of the volumes lost energy"


@_not_kept("a slow clock gathers energy by itself, the pull as good as off")
def test_it_is_the_pull_that_gathers_it_a_hundred_times_weaker_and_the_fuller_half_gains_nothing():
    held = sum(record["fuller_gain"] for record in _of("held"))
    gentle = sum(record["fuller_gain"] for record in _of("gentle"))
    assert held > 0 and gentle < held / 10, "without the pull the fuller half levels out; it gathers nothing"


@pytest.mark.parametrize("name", [
    pytest.param(name, marks=[_not_kept("the pull turns a box over half a turn a step")] if name in ("default", "uneven") else [])
    for name in sorted(RUNS)])
def test_the_pull_never_turns_a_box_by_more_than_half_a_turn_in_one_step(name):
    turns = [record["largest_turn"] for record in _of(name)]
    assert max(turns) < math.pi, f"largest {max(turns)} rad at step {turns.index(max(turns)) + 1}"
