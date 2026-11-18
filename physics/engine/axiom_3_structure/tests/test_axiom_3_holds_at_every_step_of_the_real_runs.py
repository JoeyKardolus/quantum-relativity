"""Axiom 3 on the real runs: the pattern never breaks, minus wraps every plus in one web, size is energy alone."""
from __future__ import annotations

import pytest

from physics.engine.tests.conftest import RUNS, first_bad, needs_gpu, run_records

pytestmark = needs_gpu

EVERY_RUN = pytest.mark.parametrize("name", sorted(RUNS))

BREAKS = ("touching", "bare", "one_sided", "dead_links", "loose_ways", "missing_ways", "shared_mouths", "flickered",
          "lone_joints")


@EVERY_RUN
@pytest.mark.parametrize("key", BREAKS)
def test_the_pattern_never_breaks(name, key):
    counts = [record["pattern"][key] for record in run_records(name)]
    assert not any(counts), first_bad(counts)


@EVERY_RUN
def test_minus_forms_one_web_of_joints_that_touch_joints(name):
    pieces = [record["pattern"]["web_pieces"] for record in run_records(name)]
    assert set(pieces) == {1}, f"web pieces per step, first not 1 at step {[p != 1 for p in pieces].index(True) + 1}"


@EVERY_RUN
@pytest.mark.parametrize("key", ["wrong_lengths", "dead_lengths", "dead_boxes"])
def test_every_size_is_the_law_of_the_cell_s_own_energy_and_a_dead_block_is_never_used(name, key):
    counts = [record["axiom_3_structure"][key] for record in run_records(name)]
    assert not any(counts), first_bad(counts)


def test_the_pattern_is_put_to_the_test_by_real_collapses_and_real_tears():
    assert run_records("default")[-1]["axiom_6_limits"]["collapses"] > 0
    assert run_records("uneven")[-1]["axiom_6_limits"]["tears"] > 0


def test_where_lumps_grow_the_ways_to_them_get_shorter_measured_against_the_average_way():
    records = [record["axiom_3_structure"] for record in run_records("held")]
    assert records[-1]["lump_share"] < records[0]["lump_share"], "no lump grew"
    assert records[-1]["warp"] < records[0]["warp"] < 1.0
