"""On a real universe the fit's measures agree with the engine's own books and grid measure, before and after collapses."""
import numpy as np
import pytest

from physics.engine import Settings, measure_grid, new_universe, read_books, step
from physics.fit.measures.definitions.lump_marks import LUMP_FACTOR
from physics.fit.measures.logic.gap_between_lumps import gap_between_lumps
from physics.fit.measures.logic.held_share import held_share
from physics.fit.measures.logic.measure_cells import measure_cells
from physics.fit.measures.logic.read_universe import read_universe
from physics.fit.measures.logic.take_snapshot import take_snapshot
from physics.fit.tests.needs_gpu import needs_gpu

_SETTINGS = Settings(push_strength=40.0, tear_length=200.0, unevenness=0.9)


def _agrees(universe, measured):
    books, grid = read_books(universe), measure_grid(universe)
    assert (measured["plus"], measured["minus"], measured["amount"]) == (books["plus"], books["minus"], books["amount"])
    assert measured["size"] == pytest.approx(grid["size"], rel=1e-9)
    assert (measured["volumes"], measured["joints"], measured["endless"]) == (grid["volumes"], grid["joints"], grid["endless"])


@needs_gpu
def test_a_very_uneven_start_has_lumps_a_gap_between_them_and_every_count_the_engines():
    universe = new_universe(_SETTINGS, 8, 1)
    cells = read_universe(universe)
    measured = measure_cells(cells, _SETTINGS.tear_length)
    _agrees(universe, measured)
    volumes = cells["held"][cells["kind"] == 1]
    assert measured["alive"] == 1.0 and measured["spent_share"] == 0.0 and 0.0 < measured["observer_rate"] < 1.0
    assert measured["lump_volumes"] == int((volumes >= LUMP_FACTOR * volumes.mean()).sum()) >= 2
    assert measured["lump_share"] == pytest.approx(volumes[volumes >= LUMP_FACTOR * volumes.mean()].sum() / volumes.sum())
    shot = take_snapshot(cells)
    assert held_share([shot] * 4, cells["fell_into"]) == pytest.approx(measured["lump_share"]), "nothing moved: every lump held"
    gaps = gap_between_lumps([shot] * 3)
    if measured["lumps"] >= 2:
        shortest_joint = cells["length"][cells["kind"] == -1].min()
        assert gaps[0] == gaps[1] == gaps[2] >= shortest_joint, "at least one joint lies between two lumps"
    else:
        assert gaps == [None] * 3


@needs_gpu
def test_after_collapses_the_volumes_alive_are_the_volumes_left_and_the_books_still_agree():
    universe = new_universe(_SETTINGS, 8, 1)
    for _ in range(8):
        step(universe, _SETTINGS)
    cells = read_universe(universe)
    measured = measure_cells(cells, _SETTINGS.tear_length)
    _agrees(universe, measured)
    assert measured["collapses"] == universe.collapses > 0 and measured["tears"] == 0
    assert measured["alive"] == measured["volumes"] / 64 < 1.0
    fallen = np.flatnonzero(cells["fell_into"] >= 0)
    assert len(fallen) == 64 - measured["volumes"] and (cells["kind"][fallen] == 0).all()
