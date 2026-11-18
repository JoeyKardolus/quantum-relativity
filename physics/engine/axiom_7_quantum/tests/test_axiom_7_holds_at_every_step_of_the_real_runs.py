"""Axiom 7 on the real runs: outflow that meets a fork splits by itself, its halves go on, friction takes its momentum.

Nothing is named or staged: the counts are read off what the engine's own step did.
"""
from __future__ import annotations

import pytest

from physics.engine.tests.conftest import RUNS, first_bad, needs_gpu, run_records

pytestmark = needs_gpu

EVERY_RUN = pytest.mark.parametrize("name", sorted(RUNS))
STANDING = pytest.mark.parametrize("name", ["gentle", "held"])


def _of(name: str) -> list:
    return [record["axiom_7_quantum"] for record in run_records(name)]


@EVERY_RUN
@pytest.mark.parametrize("key", ["parts_wrong", "not_whole", "over_claimed", "fractions"])
def test_every_split_keeps_its_halves_its_pointer_and_no_speed_anywhere(name, key):
    if (name, key) == ("collapsing", "not_whole"):
        pytest.xfail("not kept: a collapse some steps after the start takes quanta that halves of open splits "
                     "still claim; 4 splits are not whole at step 9 (axiom 7, Not done)")
    counts = [record[key] for record in _of(name)]
    assert not any(counts), first_bad(counts)


@EVERY_RUN
def test_the_first_crossing_already_meets_forks_and_opens_splits_by_itself(name):
    first = _of(name)[0]
    assert first["opened"] > 0 and first["open"] + first["settled_reaction"] + first["settled_momentum"] > 0


@STANDING
def test_on_a_grid_that_stands_outflow_meets_forks_and_the_halves_go_on_in_every_step(name):
    records = _of(name)
    for step, record in enumerate(records[1:], start=2):
        assert record["opened"] + record["joined"] > 0, f"no outflow met a fork in step {step}"
        assert record["carried"] > 0, f"no half went on with its cell's fluid in step {step}"


@STANDING
def test_friction_takes_direction_off_the_open_splits_in_every_step_and_never_gives_any_back(name):
    records = _of(name)
    for step, record in enumerate(records[1:], start=2):
        assert record["cooled"] > 0, f"step {step}"
        assert 0 <= record["directed"] <= record["claimed"], f"step {step}"


@EVERY_RUN
def test_what_settles_was_all_there_to_be_drawn(name):
    if name == "collapsing":
        pytest.xfail("not kept: a collapse some steps after the start takes quanta that halves of open splits "
                     "still claim; a settling is 9,367 quanta short at step 9 (axiom 7, Not done)")
    short = [record["short_quanta"] for record in _of(name)]
    assert not any(short), first_bad(short)


@EVERY_RUN
def test_settling_changes_no_total(name):
    laws = [record["book_law"] for record in run_records(name)]
    assert all(not any(law.values()) for law in laws)
