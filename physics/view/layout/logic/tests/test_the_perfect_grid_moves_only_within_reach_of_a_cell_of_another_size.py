"""The drawn grid's corners on widths whose answer is known: the perfect grid, warped only round other sizes."""
import numpy as np

from physics.view.layout.definitions.warp_marks import LIMIT, REACH
from physics.view.layout.logic.cell_layout import cell_layout
from physics.view.layout.logic.typical_lengths import typical_lengths

SIDE = 28
TYPICAL = (10.0, 4.0)


def _even(side: int = SIDE) -> np.ndarray:
    widths = np.zeros((3, side, side, side))
    for axis in range(3):
        widths[axis] = np.where(np.arange(side) % 2 == 0, 10.0, 4.0).reshape([side if other == axis else 1 for other in range(3)])
    return widths


def test_where_every_size_is_typical_the_grid_is_exactly_the_perfect_one():
    corners, limited = cell_layout(_even(), TYPICAL)
    assert corners.shape == (SIDE + 1, SIDE + 1, SIDE + 1, 3) and not limited
    for axis in range(3):
        steps = np.diff(corners[..., axis], axis=axis)
        assert np.array_equal(np.unique(steps), [4.0, 10.0]), "every slot exactly its typical width"
        assert np.ptp(corners[..., axis], axis=tuple(other for other in range(3) if other != axis)).max() == 0.0, "straight lines"
    assert corners.min() == -98.0 and corners.max() == 98.0, "the frame: 14 volumes and 14 joints, centred"


def test_a_lump_moves_only_the_corners_within_reach_and_never_the_frame():
    even, _ = cell_layout(_even(), TYPICAL)
    widths = _even()
    widths[:, 4:7, 4:7, 4:7] *= 0.5
    warped, limited = cell_layout(widths, TYPICAL)
    assert not limited, "half the typical size is within what is drawn"
    moved = np.abs(warped - even).max(axis=-1) > 0.0
    reached = np.zeros_like(moved)
    reached[max(4 - REACH, 0):7 + REACH + 1, max(4 - REACH, 0):7 + REACH + 1, max(4 - REACH, 0):7 + REACH + 1] = True
    assert moved.any() and not (moved & ~reached).any(), "a corner further than REACH from the lump is exactly where it was"
    assert np.array_equal(warped[16:], even[16:]), "the far half of the grid is untouched, to the last bit"
    for axis in range(3):
        for end in (0, SIDE):
            assert np.array_equal(np.take(warped[..., axis], end, axis=axis), np.take(even[..., axis], end, axis=axis)), \
                "the outer frame stays put along its own axis"
    inside = np.diff(warped[..., 0], axis=0)[4:7, 5, 5]
    assert (inside < np.array([10.0, 4.0, 10.0]) * 0.75).all(), "the lump's cells are drawn smaller"
    toward = (warped - even)[8:14, 5, 5, 0]
    assert (toward < 0).all() and (np.diff(toward) > 0).all(), "the lines beyond it bend toward it, less with distance"


def test_no_edge_is_ever_shorter_than_an_eighth_of_its_regular_length_whatever_the_sizes():
    rng = np.random.default_rng(7)
    widths = _even(12) * np.exp(3.0 * rng.standard_normal((3, 12, 12, 12)))
    widths[:, 3, 3, 3], widths[:, 8, 2, 5] = 0.0, np.inf
    corners, limited = cell_layout(widths, TYPICAL)
    assert limited and np.isfinite(corners).all()
    for axis in range(3):
        regular = np.where(np.arange(12) % 2 == 0, 10.0, 4.0).reshape([12 if other == axis else 1 for other in range(3)])
        drawn = np.diff(corners[..., axis], axis=axis)
        assert (drawn >= regular / (2.0 * LIMIT) - 1e-9).all(), "corners keep their order along every line"
    one, two, three = corners[1:, :-1, :-1] - corners[:-1, :-1, :-1], corners[:-1, 1:, :-1] - corners[:-1, :-1, :-1], \
        corners[:-1, :-1, 1:] - corners[:-1, :-1, :-1]
    assert (np.einsum("...i,...i->...", one, np.cross(two, three)) > 0).all(), "no block is turned inside out"
    assert np.array_equal(corners, cell_layout(widths.copy(), TYPICAL)[0]), "the same sizes, the same corners"


def test_a_cut_slab_is_laid_out_exactly_as_the_same_layers_of_the_whole_lattice():
    widths = _even(16) * np.exp(0.4 * np.random.default_rng(3).standard_normal((3, 16, 16, 16)))
    whole, _ = cell_layout(widths, TYPICAL, flat=True)
    slab, _ = cell_layout(widths[:, :, :, 5:10], TYPICAL, first=(0, 0, 5), places=16, flat=True, trim=((0, 0), (0, 0), (2, 2)))
    assert slab.shape == (17, 17, 2, 3) and np.array_equal(slab, whole[:, :, 7:9]), "one layer costs one layer"
    assert np.ptp(whole[..., 2], axis=(0, 1)).max() == 0.0, "a slice is a flat cut"


def test_the_typical_lengths_are_the_medians_so_a_lump_does_not_move_them():
    kind = np.array([1, 1, 1, 1, 1, -1, -1, -1, 0, 1])
    length = np.array([10.0, 10.5, 11.0, 0.1, np.inf, 4.0, 5.0, 0.2, 99.0, 3.0])
    assert typical_lengths(kind, length, 9) == (10.25, 4.0), "ids past the lattice, gone cells and endless ones are left out"
    assert typical_lengths(np.zeros(4, dtype=int), np.ones(4), 4) == (1.0, 1.0)
