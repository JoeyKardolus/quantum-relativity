"""Energy volumes are red, joints blue, and opacity is a fixed scale of packing: the same packing, the same colour, in any frame."""
import numpy as np

from physics.view import cell_colours, cell_pieces, slice_picture
from physics.view.cell_colours import JOINT_BLUE, SOLID, VOLUME_RED, WHOLE_POWER
from physics.view.tests.grid_cases import uneven_lattice


def test_kind_sets_the_hue_and_packing_only_the_opacity():
    packing = np.array([0.05, 0.3, 1.0, 2.0, 3.0, SOLID])
    red = cell_colours(np.ones(6, dtype=int), packing).astype(int)
    blue = cell_colours(-np.ones(6, dtype=int), packing).astype(int)
    assert (red[:, :3] == VOLUME_RED).all() and (blue[:, :3] == JOINT_BLUE).all(), "the hue never moves with packing"
    assert (np.diff(red[:, 3]) > 0).all() and np.array_equal(red[:, 3], blue[:, 3]), "more packed is more solid"
    assert red[-1, 3] == 255, "solid at SOLID times the start's mean packing"
    assert 38 <= red[2, 3] <= 77, "the even start is faint but visible in a slice"
    assert np.allclose(red[:, 3] / 255.0, packing / SOLID, atol=0.003), "a straight line on a log-log scale"
    assert cell_colours(np.array([1]), np.array([0.0]))[0, 3] == 0, "an emptied cell is fully transparent"


def test_the_scale_is_fixed_and_clipped_never_stretched_to_the_frame():
    alone = cell_colours(np.array([1]), np.array([1.3]))
    among = cell_colours(np.array([1, 1, -1, 1]), np.array([1.3, 900.0, 0.001, 1e-9]))
    assert np.array_equal(alone[0], among[0])
    assert among[1].tolist() == [*VOLUME_RED, 255], "past the scale's top it stays solid"
    assert among[2].tolist() == [*JOINT_BLUE, 0] and among[3, 3] == 0


def test_in_the_whole_grid_the_even_start_is_nearly_clear_and_a_lump_stays_solid():
    opacity = (cell_colours(np.ones(3, dtype=int), np.array([1.0, 3.0, SOLID]))[:, 3] / 255.0) ** WHOLE_POWER
    through_32_places = 1.0 - (1.0 - opacity[0]) ** 64
    assert 0.1 < through_32_places < 0.4, "32 places of even cells still show what lies behind them"
    assert opacity[1] > 0.25 and opacity[2] == 1.0


def test_a_slice_picture_shows_a_packed_cell_solid_and_an_emptied_one_as_the_black_ground():
    kind, length, packing = uneven_lattice(8, spread=0.0)
    packing = np.ones_like(packing)
    packed, emptied = 4 * 64 + 4 * 8 + 4, 2 * 64 + 2 * 8 + 4
    packing[packed], packing[emptied] = 50.0, 0.0
    pieces = cell_pieces(8, kind, length, packing)
    picture = slice_picture(pieces, 4, pieces.low, pieces.high, width=440).astype(int)
    reds = picture[(picture[..., 0] > picture[..., 2])]
    assert (reds == VOLUME_RED).all(axis=1).any(), "the packed volume is drawn solid"
    faint = reds[(reds != VOLUME_RED).any(axis=1)]
    assert len(faint) and (faint[:, 0] < 80).all() and (faint[:, 0] > 40).all(), "the even volumes are faint"
    assert np.allclose(faint[:, 1] / faint[:, 0], VOLUME_RED[1] / VOLUME_RED[0], atol=0.03), "and of the same hue"
    place = np.argwhere(pieces.cell == emptied)[0, 0]
    middle = pieces.corners[place].mean(axis=0)
    scale = 440 / (pieces.high[0] - pieces.low[0])
    column, row = int((middle[0] - pieces.low[0]) * scale), picture.shape[0] - int((middle[1] - pieces.low[1]) * scale)
    assert picture[row, column].sum() == 0, "the emptied volume has faded away"
