"""The source as a whole on real engine runs (8 cells a side, seed 1; needs CUDA).

Two runs: the defaults, where joints collapse and volumes merge, and the engine's own
uneven run (`unevenness` 0.3, `push_strength` 40), where volumes tear. After every step
the grid is laid out and checked.
"""
import numpy as np
import pytest

from physics.view.tests.gpu_cases import need_cuda

RUNS = {"collapses": ({}, 4), "tears": ({"unevenness": 0.3, "push_strength": 40.0}, 5)}


@pytest.mark.parametrize("name", list(RUNS))
def test_every_living_cell_is_drawn_once_and_stays_in_its_slot(name):
    need_cuda()
    from physics.view.source.logic.cells_source import CellsSource
    overrides, steps = RUNS[name]
    source = CellsSource(8, seed=1, overrides=overrides)
    where = {}
    for _ in range(steps + 1):
        pieces = source.view()
        kind = source.state.host("kind")
        assert sorted(pieces.cell.tolist()) == np.flatnonzero(kind != 0).tolist(), "every living cell once, no other"
        for cell, place in zip(pieces.cell.tolist(), map(tuple, pieces.place.tolist())):
            assert where.setdefault(cell, place) == place, f"cell {cell} moved from {where[cell]} to {place}"
        assert np.isfinite(pieces.corners).all() and np.isfinite(pieces.packing).all()
        source.step()
    stats = source.stats()
    assert stats[name] > 0, f"the run must have {name}"
    assert stats["volumes"] + stats["joints"] == int((source.state.host("kind") != 0).sum())


def test_a_reset_goes_back_to_the_same_start():
    need_cuda()
    from physics.view.source.logic.cells_source import CellsSource
    source = CellsSource(8, seed=1)
    start = source.view()
    source.step()
    source.step()
    assert source.steps == 2 and source.stats()["collapses"] > 0
    source.reset()
    again = source.view()
    assert source.steps == 0 and np.array_equal(start.corners, again.corners) and np.array_equal(start.packing, again.packing)
