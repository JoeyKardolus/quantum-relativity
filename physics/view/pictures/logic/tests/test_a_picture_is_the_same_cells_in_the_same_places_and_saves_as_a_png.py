"""Painting, writing and saving on a plain-array source (no engine, no OpenGL)."""
import struct
import zlib

import numpy as np
import pytest

from physics.view.colours.definitions.palette import VOLUME_RED
from physics.view.layout.logic.cell_pieces import cell_pieces
from physics.view.pictures.logic.fill_triangle import fill_triangle
from physics.view.pictures.logic.saved_frames import saved_frames
from physics.view.pictures.logic.slice_of_the_middle import slice_of_the_middle
from physics.view.pictures.logic.slice_picture import slice_picture
from physics.view.pictures.logic.write_png import write_png
from physics.view.tests.grid_cases import FakeSource, uneven_lattice


def test_a_triangle_paints_the_pixels_whose_centres_lie_in_it_whichever_way_it_winds():
    for corners in ([[0, 0], [8, 0], [0, 8]], [[0, 0], [0, 8], [8, 0]]):
        picture = np.zeros((10, 10, 3), dtype=np.uint8)
        fill_triangle(picture, np.array(corners, dtype=float), np.array([9, 8, 7]))
        painted = (picture == [9, 8, 7]).all(axis=2)
        assert painted.sum() == 36, "the 8 + 7 + ... + 1 pixel centres on or under the diagonal"
        assert painted[0, :8].all() and not painted[0, 8] and painted[7, 0] and not painted[7, 1]
    outside = np.zeros((4, 4, 3), dtype=np.uint8)
    fill_triangle(outside, np.array([[10.0, 10.0], [20.0, 10.0], [10.0, 20.0]]), np.array([1, 1, 1]))
    assert outside.sum() == 0


def test_a_written_png_reads_back_as_the_same_picture(tmp_path):
    picture = np.random.default_rng(5).integers(0, 256, size=(7, 5, 3), dtype=np.uint8)
    path = tmp_path / "one.png"
    assert write_png(path, picture) == path.stat().st_size
    data = path.read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n" and struct.unpack(">II", data[16:24]) == (5, 7) and data[24:26] == bytes([8, 2])
    length = struct.unpack(">I", data[33:37])[0]
    assert data[37:41] == b"IDAT"
    rows = np.frombuffer(zlib.decompress(data[41:41 + length]), dtype=np.uint8).reshape(7, 16)
    assert (rows[:, 0] == 0).all() and np.array_equal(rows[:, 1:].reshape(7, 5, 3), picture)
    with pytest.raises(ValueError):
        write_png(path, np.zeros((4, 4), dtype=np.uint8))


def test_a_slice_picture_is_the_same_cells_in_the_same_places():
    source = FakeSource()
    pieces = source.view()
    low, high = pieces.low - 2.0, pieces.high + 2.0
    picture = slice_picture(pieces, 4, low, high, width=400)
    assert picture.shape[1] == 400 and picture.dtype == np.uint8
    assert np.array_equal(picture, slice_picture(source.view(), 4, low, high, width=400)), "the same grid, the same picture"
    red = (picture[..., 0] > picture[..., 2]) & (picture[..., 0] > 10)
    blue = (picture[..., 2] > picture[..., 0]) & (picture[..., 2] > 10)
    black = picture.sum(axis=2) == 0
    assert red.sum() + blue.sum() + black.sum() == picture[..., 0].size, "only reds, blues and the black round the grid"
    assert 0.6 < red.sum() / (red.sum() + blue.sum()) < 0.95, "this lattice's volumes are ten times its joints"
    assert black[0].all() and black[:, 0].all() and not black[200, 40:360].any(), "no gap inside the grid"


def test_the_slice_of_the_middle_is_the_slice_at_the_middle_layer():
    pieces = FakeSource().view()
    assert np.array_equal(slice_of_the_middle(pieces, pieces.low, pieces.high),
                          slice_picture(pieces, 4, pieces.low, pieces.high))


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
    assert len(faint) and (faint == np.round(np.array(VOLUME_RED) * 64 / 255)).all(), "the even volumes at a quarter"
    place = np.argwhere(pieces.cell == emptied)[0, 0]
    middle = pieces.corners[place].mean(axis=0)
    scale = 440 / (pieces.high[0] - pieces.low[0])
    column, row = int((middle[0] - pieces.low[0]) * scale), picture.shape[0] - int((middle[1] - pieces.low[1]) * scale)
    assert picture[row, column].sum() == 0, "the emptied volume has faded away"


def test_saved_frames_step_the_source_and_hand_every_picture_the_same_view(tmp_path):
    source, views = FakeSource(), []

    def picture(pieces, low, high):
        views.append((low.copy(), high.copy()))
        return slice_of_the_middle(pieces, low, high)

    frames = saved_frames(source, 5, 2, tmp_path, picture)
    assert [frame["step"] for frame in frames] == [0, 2, 4, 5] and source.steps == 5
    assert all(np.array_equal(view[0], views[0][0]) and np.array_equal(view[1], views[0][1]) for view in views)
    assert (views[0][0] < source.view().low).all(), "a margin round the start's grid"
    assert sorted(path.name for path in tmp_path.iterdir()) == [f"step_0000{step}.png" for step in (0, 2, 4, 5)]
    assert frames[0]["bytes"] == (tmp_path / "step_00000.png").stat().st_size and frames[0]["volumes"] == 64
    assert (tmp_path / "step_00000.png").read_bytes() != (tmp_path / "step_00005.png").read_bytes()
    with pytest.raises(ValueError):
        saved_frames(source, 5, 0, tmp_path, picture)
