"""The layout draws a perfect A B A B pattern: no cell overlaps another, no gap, every cell in its place in the pattern.

Plain arrays, no engine. A lattice whose cells' lengths differ by 30% (three times the
default start's unevenness), and an even one where every size is exact.
"""
import numpy as np

from physics.view import cell_pieces
from physics.view.tests.grid_cases import inside, slice_faces, uneven_lattice

SIDE = 8


def test_no_point_lies_in_two_cells_and_none_in_no_cell():
    kind, length, packing = uneven_lattice(SIDE)
    pieces = cell_pieces(SIDE, kind, length, packing)
    assert len(pieces.cell) == SIDE ** 3
    points = np.random.default_rng(2).uniform(-30.0, 30.0, size=(4000, 2))
    for layer in range(SIDE):
        faces, _ = slice_faces(pieces, layer)
        assert len(faces) == SIDE * SIDE
        hits = inside(points, faces)
        assert hits.max() == 1, "a point lies inside two cells: cells overlap"
        hull = np.abs(points).max(axis=1) < 12.0
        assert (hits[hull] == 1).mean() > 0.999, "a gap inside the grid (only points exactly on an edge may miss)"


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


def test_a_cell_with_more_energy_is_drawn_smaller_and_the_grid_closes_in_round_it():
    kind, length, packing = uneven_lattice(SIDE, spread=0.0)
    even = cell_pieces(SIDE, kind, length, packing)
    lump = (2 * SIDE + 2) * SIDE + 2
    length[lump] = 2.5
    warped = cell_pieces(SIDE, kind, length, packing)
    middle = [(warped.corners[:, [c for c in range(8) if c >> axis & 1], axis].mean(axis=1)
               - warped.corners[:, [c for c in range(8) if not c >> axis & 1], axis].mean(axis=1)) for axis in range(3)]
    middle = np.stack(middle, axis=1)
    volumes = np.flatnonzero(warped.kind == 1)
    assert (middle[lump] < middle[volumes[volumes != lump]].min(axis=0)).all(), "the lump is the smallest volume drawn"
    assert (middle > 0).all()
    assert ((warped.high - warped.low) < (even.high - even.low)).all(), "the whole grid is smaller by what the lump lost"
    far = (np.abs(warped.place - 2) > 3).all(axis=1) & (warped.kind == 1)
    assert np.abs(middle[far] / 10.0 - 1.0).max() < 0.05, "volumes far from the lump keep their size"
