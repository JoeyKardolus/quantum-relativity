"""The corner fit on widths whose answer is known."""
import numpy as np

from physics.view.layout.logic.cell_layout import cell_layout


def _widths(side: int, lengths) -> np.ndarray:
    widths = np.zeros((3, side, side, side))
    for axis in range(3):
        shape = [side if other == axis else 1 for other in range(3)]
        widths[axis] = np.asarray(lengths, dtype=float).reshape(shape)
    return widths


def test_layers_that_agree_are_drawn_exactly_their_width_and_centred_on_nothing():
    lengths = [10.0, 1.0, 4.0, 300.0]
    corners = cell_layout(_widths(4, lengths))
    assert corners.shape == (5, 5, 5, 3)
    for axis in range(3):
        steps = np.diff(corners[..., axis], axis=axis)
        assert np.allclose(steps, np.asarray(lengths).reshape([4 if other == axis else 1 for other in range(3)]),
                           rtol=1e-6), "every slot exactly as wide as asked"
        others = tuple(other for other in range(3) if other != axis)
        assert np.ptp(corners[..., axis], axis=others).max() < 1e-4, "every edge straight along its axis"
    assert np.allclose(corners.reshape(-1, 3).mean(axis=0), 0.0, atol=1e-6)
    assert np.allclose(np.ptp(corners.reshape(-1, 3), axis=0), 315.0, rtol=1e-6)


def test_the_same_widths_give_the_same_corners_and_an_empty_slot_closes_up_without_folding():
    widths = _widths(4, [10.0, 1.0, 10.0, 1.0])
    widths[:, 2, 2, 2] = 0.0
    one, two = cell_layout(widths), cell_layout(widths.copy())
    assert np.array_equal(one, two)
    for axis in range(3):
        assert (np.diff(one[..., axis], axis=axis) > 0).all(), "corners keep their order along every row"
    assert np.diff(one[..., 0], axis=0)[2, 2, 2] < 5.0, "the emptied slot is drawn narrower than the 10 it held"
