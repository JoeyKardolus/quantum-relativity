"""The engine's fresh start read as pieces and stats: 8 cells a side, seed 1, defaults (needs CUDA)."""
import numpy as np
import pytest

from physics.view.layout.definitions import kinds
from physics.view.tests.gpu_cases import need_cuda

SIDE = 8


@pytest.fixture(scope="module")
def source():
    need_cuda()
    from physics.view.source.logic.cells_source import CellsSource
    return CellsSource(SIDE, seed=1)


def test_the_viewers_kind_numbers_are_the_engines():
    need_cuda()
    import physics.engine as engine
    assert (kinds.ENERGY_VOLUME, kinds.JOINT, kinds.NO_CELL) == (engine.ENERGY_VOLUME, engine.JOINT, engine.NO_CELL)


def test_every_start_cell_is_drawn_once_in_its_own_place_with_its_engine_kind(source):
    pieces = source.view()
    assert len(pieces.cell) == SIDE ** 3 and pieces.cell.tolist() == list(range(SIDE ** 3))
    assert np.array_equal(pieces.kind, source.state.host("kind"))
    assert ((pieces.place % 2 == 0).all(axis=1) == (pieces.kind == 1)).all(), "volumes where all three indices are even"


def test_packing_is_read_against_the_starts_own_mean_of_each_kind(source):
    pieces = source.view()
    for kind in (1, -1):
        assert abs(float(pieces.packing[pieces.kind == kind].mean()) - 1.0) < 1e-5
    packing = source.state.host("packing")
    assert np.allclose(pieces.packing, packing / np.where(pieces.kind == 1, source.marks[1], source.marks[-1]), rtol=1e-5)


def test_cells_are_drawn_at_the_engines_lengths_joints_far_longer_than_volumes(source):
    pieces = source.view()
    length = source.state.host("length").astype(np.float64)
    spans = pieces.corners.max(axis=1) - pieces.corners.min(axis=1)
    odd = pieces.place % 2 == 1
    own = np.where(odd | (pieces.kind == 1)[:, None], length[:, None], np.nan)
    miss = np.abs(spans / own - 1.0)
    assert np.nanmax(miss[pieces.kind == -1]) < 0.02, "a joint is drawn its own length along its odd axes"
    assert np.nanmean(miss[pieces.kind == 1]) < 0.25, "a volume about its own length (the fit shares the mismatch)"
    assert length[pieces.kind == -1].min() > 20 * length[pieces.kind == 1].max(), "one size law: less energy, longer"
    assert np.allclose(pieces.high - pieces.low, np.nansum(own.reshape(SIDE, SIDE, SIDE, 3)[:, 0, 0, 0]), rtol=0.05)


def test_the_stats_are_the_engines_own_counts(source):
    from physics.engine import measure_grid, read_books
    stats = source.stats()
    measured = measure_grid(source.state)
    assert stats["volumes"] == (SIDE // 2) ** 3 and stats["joints"] == SIDE ** 3 - (SIDE // 2) ** 3
    assert stats["collapses"] == 0 and stats["tears"] == 0 and stats["endless"] == 0
    assert stats["size"] == measured["size"] and stats["size_growth"] == 1.0
    assert stats["amount"] == read_books(source.state)["amount"] and isinstance(stats["amount"], int)


def test_a_setting_the_engine_does_not_have_is_refused():
    need_cuda()
    from physics.view.source.logic.cells_source import CellsSource
    with pytest.raises(TypeError):
        CellsSource(SIDE, overrides={"no_such_setting": 1.0})
