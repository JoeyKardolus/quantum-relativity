"""Axiom 1 on the real runs: exact books, whole signed boxes, and a fluid that moves in every cell and between cells."""
from __future__ import annotations

import pytest

from physics.engine.axiom_1_energy.definitions.energy_arrays import ARRAYS
from physics.engine.tests.conftest import RUNS, first_bad, needs_gpu, run_records

pytestmark = needs_gpu

EVERY_RUN = pytest.mark.parametrize("name", sorted(RUNS))


def _of(name: str) -> list:
    return [record["axiom_1_energy"] for record in run_records(name)]


def test_quanta_are_whole_by_type():
    assert ARRAYS["quanta"][0] == "int32" and ARRAYS["cell_quanta"][0] == "int64"


@EVERY_RUN
@pytest.mark.parametrize("key", ["book_amount", "book_plus", "book_minus", "book_taken", "wrong_boxes",
                                 "wrong_cells"])
def test_every_change_of_the_total_is_booked_and_every_box_holds_a_whole_number_of_its_cell_s_sign(name, key):
    counts = [record[key] for record in _of(name)]
    assert not any(counts), first_bad(counts)


@EVERY_RUN
def test_the_universe_holds_energy_and_no_cell_that_can_be_judged_stands_still_on_its_own_clock(name):
    records = _of(name)
    assert all(record["plus"] > 0 and record["minus"] > 0 for record in records), "the universe ran empty"
    shares = [record["still_share"] for record in records]
    assert max(shares) == 0.0, f"largest share standing still: {max(shares)} at step {shares.index(max(shares)) + 1}"


@EVERY_RUN
def test_while_any_two_volumes_share_a_way_energy_crosses_between_cells_in_every_step(name):
    stuck = [step for step, record in enumerate(_of(name), start=1)
             if record["acting_ways"] > 0 and record["crossed_share"] == 0.0]
    assert not stuck, f"nothing crossed although ways were open at steps {stuck[:10]} ({len(stuck)} steps)"


def test_on_a_grid_that_stands_more_than_a_hundredth_of_the_plus_energy_crosses_in_every_step():
    shares = [record["crossed_share"] for record in _of("held")]
    assert min(shares) > 0.01, f"least {min(shares)} at step {shares.index(min(shares)) + 1}"


def test_the_total_is_not_fixed_squeezed_joints_make_and_collapses_take():
    last = _of("default")[-1]
    assert last["made_total"] > 0 and last["taken_joint"] > 0 and last["taken_volumes"] > 0
