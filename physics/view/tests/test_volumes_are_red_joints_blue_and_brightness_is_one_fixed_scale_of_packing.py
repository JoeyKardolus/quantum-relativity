"""Energy volumes are red, joints blue, and brightness is a fixed scale of packing: the same packing, the same colour, in any frame."""
import numpy as np

from physics.view import cell_colours
from physics.view.cell_colours import DIMMEST, JOINT_BLUE, OCTAVES, VOLUME_RED


def test_kind_sets_the_hue_and_packing_the_brightness():
    packing = np.array([2.0 ** -OCTAVES, 1.0, 2.0 ** OCTAVES])
    red = cell_colours(np.array([1, 1, 1]), packing).astype(int)
    blue = cell_colours(np.array([-1, -1, -1]), packing).astype(int)
    assert red[2].tolist() == list(VOLUME_RED) and blue[2].tolist() == list(JOINT_BLUE)
    assert (red[:, 0] > red[:, 1]).all() and (red[:, 0] > red[:, 2]).all(), "volumes are red"
    assert (blue[:, 2] > blue[:, 0]).all() and (blue[:, 2] > blue[:, 1]).all(), "joints are blue"
    assert red[0, 0] < red[1, 0] < red[2, 0] and blue[0, 2] < blue[1, 2] < blue[2, 2], "more packed is brighter"
    assert abs(red[0, 0] - DIMMEST * 255) <= 1 and abs(red[1, 0] - (1 + DIMMEST) / 2 * 255) <= 1


def test_the_scale_is_fixed_and_clipped_never_stretched_to_the_frame():
    alone = cell_colours(np.array([1]), np.array([1.3]))
    among = cell_colours(np.array([1, 1, -1, 1]), np.array([1.3, 900.0, 0.001, 1e-9]))
    assert np.array_equal(alone[0], among[0])
    assert among[1].tolist() == list(VOLUME_RED), "past the scale's top it stays full"
    assert np.array_equal(among[3], cell_colours(np.array([1]), np.array([2.0 ** -OCTAVES]))[0])
    assert among.min(axis=1).max() < 255 and (among.max(axis=1) >= DIMMEST * 200).all(), "no cell is ever black"
