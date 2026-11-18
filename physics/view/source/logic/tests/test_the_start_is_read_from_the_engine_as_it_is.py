"""The engine's fresh start read as pieces and stats: 8 cells a side, seed 1, defaults (needs CUDA)."""
import numpy as np
import pytest

from physics.view.layout.definitions import kinds
from physics.view.layout.logic.typical_lengths import typical_lengths
from physics.view.tests.gpu_cases import need_cuda
from physics.view.tests.grid_cases import every_layer

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


def _all(source, name):
    layers = every_layer(source, SIDE)
    order = np.argsort(np.concatenate([pieces.cell for pieces in layers]))
    return np.concatenate([getattr(pieces, name) for pieces in layers])[order]


def test_every_start_cell_is_drawn_once_in_its_own_place_with_its_engine_kind(source):
    assert _all(source, "cell").tolist() == list(range(SIDE ** 3))
    kind, place = _all(source, "kind"), _all(source, "place")
    assert np.array_equal(kind, source.state.host("kind"))
    assert ((place % 2 == 0).all(axis=1) == (kind == 1)).all(), "volumes where all three indices are even"
    assert all((pieces.place[:, 2] == layer).all() for layer, pieces in enumerate(every_layer(source, SIDE)))


def test_packing_is_read_against_the_starts_own_mean_of_each_kind(source):
    kinds, relative = _all(source, "kind"), _all(source, "packing")
    for kind in (1, -1):
        assert abs(float(relative[kinds == kind].mean()) - 1.0) < 1e-5
    packing = source.state.host("packing")
    assert np.allclose(relative, packing / np.where(kinds == 1, source.marks[1], source.marks[-1]), rtol=1e-5)
    packed = source.view()
    assert len(packed.cell) < SIDE ** 3 and (packed.packing >= 1.69).all(), "the whole grid: only cells its scale shows"


def test_the_start_is_drawn_as_the_perfect_grid_of_the_engines_typical_lengths_joints_longer_than_volumes(source):
    length = source.state.host("length").astype(np.float64)
    volume, joint = typical_lengths(source.state.host("kind"), length, SIDE ** 3)
    assert joint > 1.5 * volume, "one size law: a default joint in balance holds less than a volume, so it is longer"
    corners, place = _all(source, "corners").astype(np.float64), _all(source, "place")
    spans = (corners[:, 7] - corners[:, 0])[:, :2]
    regular = np.where(place[:, :2] % 2 == 1, joint, volume)
    assert np.abs(spans / regular - 1.0).mean() < 0.1 and (spans > regular / 8.0).all(), \
        "an uneven start is the perfect grid with every cell a little off its typical size"
    pieces = source.view(4)
    assert np.allclose(pieces.high, 0.25 * SIDE * (volume + joint), rtol=1e-5) and np.allclose(pieces.low, -pieces.high)


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
