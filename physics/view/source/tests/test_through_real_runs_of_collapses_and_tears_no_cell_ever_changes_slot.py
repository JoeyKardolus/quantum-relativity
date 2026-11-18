"""The source as a whole on real engine runs (8 cells a side, seed 1; needs CUDA).

Two runs: the defaults, where joints collapse and volumes merge, and the engine's own
uneven run (`unevenness` 0.3, `push_strength` 40), where volumes tear. After every step
the grid is laid out and checked. Then the picture itself, on the gravity scene with
two lumps (the fitted phase 1 settings, 24 places a side, 200 steps) and on the default
universe: every drawn cell has area, no two overlap, and nothing jumps between frames.
"""
import numpy as np
import pytest

from physics.view.tests.gpu_cases import need_cuda
from physics.view.tests.grid_cases import every_layer, inside, slice_faces, turns

RUNS = {"collapses": ({"collapse_thin": 1.01}, 14), "tears": ({"unevenness": 0.6, "push_strength": 40.0}, 5)}


@pytest.mark.parametrize("name", list(RUNS))
def test_every_living_cell_is_drawn_once_and_stays_in_its_slot(name):
    need_cuda()
    from physics.view.source.logic.cells_source import CellsSource
    overrides, steps = RUNS[name]
    source = CellsSource(8, seed=1, overrides=overrides)
    where = {}
    for _ in range(steps + 1):
        layers = every_layer(source, 8)
        kind = source.state.host("kind")
        cells = np.concatenate([pieces.cell for pieces in layers])
        assert sorted(cells.tolist()) == np.flatnonzero(kind != 0).tolist(), "every living cell once, no other"
        for pieces in layers:
            for cell, place in zip(pieces.cell.tolist(), map(tuple, pieces.place.tolist())):
                assert where.setdefault(cell, place) == place, f"cell {cell} moved from {where[cell]} to {place}"
            assert np.isfinite(pieces.corners).all() and np.isfinite(pieces.packing).all()
        packed = source.view()
        assert set(packed.cell.tolist()) <= set(cells.tolist()) and (packed.packing > 1.0).all(), "the whole grid: packed cells only"
        source.step()
    stats = source.stats()
    assert stats[name] > 0, f"the run must have {name}"
    assert stats["volumes"] + stats["joints"] == int((source.state.host("kind") != 0).sum())


def test_a_reset_goes_back_to_the_same_start():
    need_cuda()
    from physics.view.source.logic.cells_source import CellsSource
    source = CellsSource(8, seed=1)
    start = source.view(4)
    source.step()
    source.step()
    assert source.steps == 2
    source.reset()
    again = source.view(4)
    assert source.steps == 0 and np.array_equal(start.corners, again.corners) and np.array_equal(start.packing, again.packing)


SCENES = {"two_lumps": (24, "two_lumps", 200, 20), "default": (8, None, 40, 10)}


@pytest.mark.parametrize("name", list(SCENES))
def test_through_a_real_run_every_drawn_cell_has_area_none_overlaps_and_nothing_jumps(name):
    need_cuda()
    from physics.fit.fitted.definitions.fitted_values import FITTED
    from physics.view.source.logic.cells_source import CellsSource
    cells, scene, steps, every = SCENES[name]
    source = CellsSource(cells, seed=1, overrides=dict(FITTED["settings"]) if scene else None, scene=scene, quantum=False)
    side = source.state.side
    points_at = np.random.default_rng(8).uniform(-1.0, 1.0, size=(1500, 2))
    before, most = None, 0.0
    for step in range(steps + 1):
        if step % every == 0:
            for layer in (0, side // 2, side // 2 + 1):
                pieces = source.view(layer)
                faces, _ = slice_faces(pieces, layer)
                assert len(faces) and (turns(faces) > 0).all(), f"step {step} layer {layer}: a cell with no area or a fold"
                hits = inside(points_at * pieces.high[:2].astype(np.float64), faces)
                assert hits.max() == 1 and (hits == 1).mean() > 0.995, f"step {step} layer {layer}: overlap or gap"
                assert np.allclose(np.abs(faces).max(axis=(0, 1)), pieces.high[:2]), "the outer frame is the regular grid's"
            blocks = source.view().corners.astype(np.float64)
            edges = blocks[:, [1, 2, 4]] - blocks[:, [0]]
            assert (np.einsum("ni,ni->n", edges[:, 0], np.cross(edges[:, 1], edges[:, 2])) > 0).all(), "a block with no volume"
        middle = source.view(side // 2)
        shape = middle.corners.astype(np.float64) / float(middle.high[0])
        if before is not None and before[0].shape == shape.shape and np.array_equal(before[1], middle.cell):
            most = max(most, float(np.abs(shape - before[0]).max()))
        before = (shape, middle.cell)
        source.step()
    assert most < 0.04, f"between two steps no corner moves more than a 25th of the half frame; most was {most:.4f}"
