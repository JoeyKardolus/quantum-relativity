"""The layout draws a perfect A B A B pattern: no cell overlaps another, no gap, every cell in its place.

Plain arrays, no engine. A lattice whose cells' lengths differ by 30%, an even one
where every size is exact, one with the engine's own proportions (a joint thirty
times as long as an energy volume), one whose sizes differ thousands of times, and a
lump that moves through a larger lattice.
"""
import numpy as np

from physics.view.layout.definitions.warp_marks import LAID_OUT_MOST, REACH
from physics.view.layout.logic.cell_pieces import cell_pieces
from physics.view.tests.grid_cases import inside, slice_faces, uneven_lattice

SIDE = 8


def test_no_point_lies_in_two_cells_and_none_in_no_cell():
    kind, length, packing = uneven_lattice(SIDE)
    pieces = cell_pieces(SIDE, kind, length, packing)
    assert len(pieces.cell) == SIDE ** 3
    points = np.random.default_rng(2).uniform(pieces.low[:2], pieces.high[:2], size=(4000, 2))
    for layer in range(SIDE):
        faces, _ = slice_faces(pieces, layer)
        assert len(faces) == SIDE * SIDE
        hits = inside(points, faces)
        assert hits.max() == 1, "a point lies inside two cells: cells overlap"
        assert (hits == 1).mean() > 0.999, "a gap inside the grid (only points exactly on an edge may miss)"


def test_a_layer_laid_out_alone_is_the_same_picture_as_that_layer_of_the_whole_grid():
    kind, length, packing = uneven_lattice(SIDE, spread=0.05)
    pieces = cell_pieces(SIDE, kind, length, packing)
    assert not pieces.limited
    for layer in range(SIDE):
        faces, cells = slice_faces(pieces, layer)
        flat, alone = slice_faces(cell_pieces(SIDE, kind, length, packing, layer=layer), layer)
        assert np.array_equal(cells, alone) and np.allclose(flat, faces, atol=1e-4)


def test_sizes_thousands_of_times_apart_gone_cells_and_endless_ones_never_fold_or_overlap_a_cell():
    kind, length, packing = uneven_lattice(SIDE, seed=5, spread=3.0)
    length[[7, 100, 301]] = np.inf
    kind[[9, 210]], length[[9, 210]] = 0, 0.0
    assert length[np.isfinite(length)].max() / length[length > 0].min() > 1e4
    pieces = cell_pieces(SIDE, kind, length, packing)
    assert pieces.limited and np.isfinite(pieces.corners).all() and len(pieces.cell) == SIDE ** 3 - 2
    corners = pieces.corners.astype(np.float64)
    edges = corners[:, [1, 2, 4]] - corners[:, [0]]
    assert (np.einsum("ni,ni->n", edges[:, 0], np.cross(edges[:, 1], edges[:, 2])) > 0).all(), "every block has volume"
    points = np.random.default_rng(6).uniform(pieces.low[:2], pieces.high[:2], size=(3000, 2))
    for layer in range(SIDE):
        faces, _ = slice_faces(pieces, layer)
        ahead, back = np.roll(faces, -1, axis=1) - faces, faces - np.roll(faces, 1, axis=1)
        assert (back[..., 0] * ahead[..., 1] - back[..., 1] * ahead[..., 0] > 0).all(), "every face has area and no fold"
        assert inside(points, faces).max() == 1, "no two cells overlap"
        flat, _ = slice_faces(cell_pieces(SIDE, kind, length, packing, layer=layer), layer)
        ahead, back = np.roll(flat, -1, axis=1) - flat, flat - np.roll(flat, 1, axis=1)
        assert (back[..., 0] * ahead[..., 1] - back[..., 1] * ahead[..., 0] > 0).all() and inside(points, flat).max() == 1, \
            "the layer laid out alone does not fold or overlap either"


def test_every_cell_turns_the_same_way_and_shares_its_corners_with_its_neighbours():
    kind, length, packing = uneven_lattice(SIDE)
    pieces = cell_pieces(SIDE, kind, length, packing)
    corners = pieces.corners.astype(np.float64)
    assert (corners[:, 1, 0] > corners[:, 0, 0]).all() and (corners[:, 2, 1] > corners[:, 0, 1]).all()
    assert (corners[:, 4, 2] > corners[:, 0, 2]).all()
    for layer in range(SIDE):
        faces, _ = slice_faces(pieces, layer)
        ahead, back = np.roll(faces, -1, axis=1) - faces, faces - np.roll(faces, 1, axis=1)
        assert (back[..., 0] * ahead[..., 1] - back[..., 1] * ahead[..., 0] > 0).all(), "a folded or flipped cell"
    by_place = {tuple(place): corner for place, corner in zip(pieces.place.tolist(), pieces.corners)}
    for (one, two, three), mine in by_place.items():
        if one + 1 < SIDE:
            assert np.array_equal(mine[[1, 3, 5, 7]], by_place[(one + 1, two, three)][[0, 2, 4, 6]])
        if two + 1 < SIDE:
            assert np.array_equal(mine[[2, 3, 6, 7]], by_place[(one, two + 1, three)][[0, 1, 4, 5]])
        if three + 1 < SIDE:
            assert np.array_equal(mine[[4, 5, 6, 7]], by_place[(one, two, three + 1)][[0, 1, 2, 3]])


def test_the_pattern_order_is_kept_volume_joint_volume_joint_along_every_row():
    kind, length, packing = uneven_lattice(SIDE)
    pieces = cell_pieces(SIDE, kind, length, packing)
    centre = pieces.corners.mean(axis=1)
    place = pieces.place
    assert np.array_equal((place[:, 0] * SIDE + place[:, 1]) * SIDE + place[:, 2], pieces.cell)
    for axis in range(3):
        others = [other for other in range(3) if other != axis]
        for row in np.unique(place[:, others], axis=0):
            mine = np.flatnonzero((place[:, others] == row).all(axis=1))
            mine = mine[np.argsort(place[mine, axis])]
            assert (np.diff(centre[mine, axis]) > 0).all(), "cells out of order along a row"
            if (row % 2 == 0).all():
                assert pieces.kind[mine].tolist() == [1, -1] * (SIDE // 2)
            else:
                assert (pieces.kind[mine] == -1).all()
    volumes = pieces.kind == 1
    assert (place[volumes] % 2 == 0).all() and volumes.sum() == (SIDE // 2) ** 3


def test_in_an_even_grid_every_cell_is_drawn_exactly_its_own_size():
    kind, length, packing = uneven_lattice(SIDE, spread=0.0)
    pieces = cell_pieces(SIDE, kind, length, packing)
    spans = pieces.corners.max(axis=1) - pieces.corners.min(axis=1)
    odd = pieces.place % 2 == 1
    assert np.allclose(spans[pieces.kind == 1], 10.0, atol=1e-4)
    assert np.allclose(spans[odd], 1.0, atol=1e-4), "a joint is as thick as its length across"
    assert np.allclose(spans[~odd], 10.0, atol=1e-4), "and as long as the volumes it lies beside"
    assert np.allclose(pieces.high - pieces.low, 4 * 11.0, atol=1e-4), "the grid is as wide as the sum of its cells"


def test_with_the_engines_own_sizes_the_joints_are_the_fat_cells_and_nothing_is_swapped():
    kind, length, packing = uneven_lattice(SIDE, spread=0.0)
    length = np.where(kind == 1, 10.0, 300.0)
    pieces = cell_pieces(SIDE, kind, length, packing)
    spans = pieces.corners.max(axis=1) - pieces.corners.min(axis=1)
    odd = pieces.place % 2 == 1
    assert np.allclose(spans[odd], 300.0, rtol=1e-5) and np.allclose(spans[~odd], 10.0, rtol=1e-4)
    assert np.allclose(pieces.high - pieces.low, 4 * 310.0, rtol=1e-5)
    volume = np.prod(spans, axis=1)
    assert volume[pieces.kind == 1].sum() / volume.sum() < 1e-4, "the energy volumes are specks between the joints"


def test_a_lump_is_drawn_smaller_and_solid_and_everything_out_of_its_reach_stays_exactly_where_it_was():
    side = 28
    kind, length, packing = uneven_lattice(side, spread=0.0)
    even = cell_pieces(side, kind, length, packing)
    place = np.indices((side, side, side)).reshape(3, -1).T
    lump = ((place >= 4) & (place <= 6)).all(axis=1)
    length[lump] *= 0.4
    warped = cell_pieces(side, kind, length, packing)
    assert np.array_equal(warped.low, even.low) and np.array_equal(warped.high, even.high), "the outer frame stays put"
    far = (warped.place > 6 + REACH + 1).any(axis=1)
    assert far.sum() > side ** 3 // 2 and np.array_equal(warped.corners[far], even.corners[far]), \
        "a cell out of the lump's reach is drawn exactly where it was"
    spans = lambda pieces: pieces.corners[:, 7] - pieces.corners[:, 0]
    assert (spans(warped)[lump] < spans(even)[lump]).all(), "every cell of the lump is drawn smaller"
    assert (spans(warped)[lump] / spans(even)[lump]).mean() < 0.85
    beside = (warped.place[:, 0] >= 8) & (warped.place[:, 0] <= 12) & (warped.place[:, 1:] == 5).all(axis=1)
    assert beside.sum() == 5 and (spans(warped)[beside, 0] > spans(even)[beside, 0]).all(), \
        "the cells beyond it along its row take up the room it gave"
    moved = cell_pieces(side, kind, np.roll(length.reshape(side, side, side), 2, axis=0).reshape(-1), packing)
    still = (moved.place > 8 + REACH + 1).any(axis=1)
    assert np.array_equal(moved.corners[still], warped.corners[still]), "when the lump moves, the far grid does not"


def test_only_the_wanted_cells_are_laid_out_and_they_lie_where_the_whole_layout_puts_them():
    side = 28
    kind, length, packing = uneven_lattice(side, spread=0.1)
    place = np.indices((side, side, side)).reshape(3, -1).T
    only = ((place >= 20) & (place <= 23)).all(axis=1)
    whole, some = cell_pieces(side, kind, length, packing), cell_pieces(side, kind, length, packing, only=only)
    assert some.cell.tolist() == np.flatnonzero(only).tolist() and some.left_out == 0
    assert np.array_equal(some.corners, whole.corners[only]), "the box round them is laid out exactly as the whole is"
    assert np.array_equal(some.low, whole.low) and np.array_equal(some.high, whole.high)
    none = cell_pieces(side, kind, length, packing, only=np.zeros(side ** 3, dtype=bool))
    assert len(none.cell) == 0 and none.left_out == 0 and none.corners.shape == (0, 8, 3)


def test_wanted_cells_too_spread_to_lay_out_are_counted_not_drawn():
    side = 128
    kind = np.ones(side ** 3, dtype=np.int8)
    only = np.zeros(side ** 3, dtype=bool)
    only[[0, side ** 3 - 1]] = True
    spread = cell_pieces(side, kind, np.full(side ** 3, 5.0, dtype=np.float32), np.ones(side ** 3, dtype=np.float32), only=only)
    assert len(spread.cell) == 0 and spread.left_out == 2 and side ** 3 > LAID_OUT_MOST
    one_layer = cell_pieces(side, kind, np.full(side ** 3, 5.0, dtype=np.float32), np.ones(side ** 3, dtype=np.float32), layer=64)
    assert len(one_layer.cell) == side * side and np.ptp(one_layer.corners[:, :4, 2]) == 0.0, "a layer is always laid out"
