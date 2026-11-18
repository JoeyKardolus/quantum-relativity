"""Axiom 4 on the real runs: energy crosses between volumes only through joints, and only a joint's friction makes heat."""
from __future__ import annotations

import pytest

from physics.engine import Settings
from physics.engine.tests.conftest import RUNS, first_bad, needs_gpu, run_records

pytestmark = needs_gpu

EVERY_RUN = pytest.mark.parametrize("name", sorted(RUNS))


def _of(name: str) -> list:
    return [record["axiom_4_crossing"] for record in run_records(name)]


@EVERY_RUN
@pytest.mark.parametrize("key", ["heat_wrong", "heated_elsewhere", "ways_without_joint", "crossed_elsewhere"])
def test_nothing_crosses_but_through_a_joint_and_heat_is_made_nowhere_else(name, key):
    counts = [record[key] for record in _of(name)]
    assert not any(counts), first_bad(counts)


@EVERY_RUN
def test_the_joints_take_the_direction_off_their_share_of_everything_that_crosses_in_every_step(name):
    friction = Settings().friction
    for step, record in enumerate(_of(name), start=1):
        assert 0 <= record["heated"] <= record["crossed"] and 0 <= record["net_crossed"] <= record["crossed"]
        if record["crossed"] >= 100_000:
            assert abs(record["heated"] / record["crossed"] - friction) < 0.01, f"step {step}"


def test_on_a_grid_that_stands_energy_really_changes_volume_in_every_step():
    records = _of("held")
    assert all(record["net_crossed"] > 0 for record in records)


def test_on_a_grid_that_stands_heat_rises_only_by_what_friction_took_and_never_falls():
    records = _of("held")
    heat = [0] + [record["heat"] for record in records]
    rises = [later - earlier for earlier, later in zip(heat, heat[1:])]
    assert all(0 <= rise <= record["heated"] for rise, record in zip(rises, records))
    assert records[-1]["heat_share"] > records[0]["heat_share"] > 0.0
