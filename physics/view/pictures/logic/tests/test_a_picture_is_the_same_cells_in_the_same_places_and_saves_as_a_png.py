"""Painting, writing and saving on a plain-array source (no engine, no OpenGL)."""
import struct
import zlib

import numpy as np
import pytest

from physics.view.colours.definitions.palette import VOLUME_RED
from physics.view.layout.logic.cell_pieces import cell_pieces
from physics.view.pictures.logic.draw_lines import draw_lines
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


def test_lines_are_blended_one_pixel_wide_and_cut_off_at_the_picture():
    picture = np.full((6, 8, 3), 100, dtype=np.uint8)
    draw_lines(picture, np.array([[1.5, 2.5], [6.5, -3.0]]), np.array([[5.5, 2.5], [6.5, 20.0]]),
               np.array([[200, 0, 0], [0, 0, 200]]), 0.5)
    assert (picture[2, 1:6] == [150, 50, 50]).all() and (picture[:, 6] == [50, 50, 150]).all()
    assert (np.delete(np.delete(picture, 6, axis=1), 2, axis=0) == 100).all(), "nothing else is touched"
    draw_lines(picture, np.zeros((0, 2)), np.zeros((0, 2)), np.zeros((0, 3)), 1.0)


def test_with_every_fill_clear_the_outlines_still_show_the_pattern_red_blue_red_blue_along_a_row():
    kind, length, _ = uneven_lattice(8, spread=0.0)
    pieces = cell_pieces(8, kind, length, np.zeros(len(kind)))
    picture = slice_picture(pieces, 4, pieces.low, pieces.high, width=880).astype(int)
    assert np.array_equal(picture, slice_picture(pieces, 4, pieces.low, pieces.high, width=880)), "the same grid, the same picture"
    red = (picture[..., 0] > 2 * picture[..., 2]) & (picture[..., 0] > 100)
    blue = (picture[..., 2] > 2 * picture[..., 0]) & (picture[..., 2] > 100)
    black = picture.sum(axis=2) == 0
    assert red.sum() + blue.sum() + black.sum() == picture[..., 0].size, "only the two hues' lines on the black ground"
    row = picture.shape[0] - 100
    hues = ["A" if red[row, column] else "B" for column in range(880) if not black[row, column]]
    runs = [hue for index, hue in enumerate(hues) if index == 0 or hues[index - 1] != hue]
    assert runs == ["A", "B"] * 4, "a row through the volumes: volume, joint, volume, joint, each outlined in its hue"
    joints_only = picture.shape[0] - 210
    assert blue[joints_only].sum() > 8 and not red[joints_only].any(), "a row between the volumes is all joints"


def test_a_slice_picture_is_the_same_cells_in_the_same_places():
    source = FakeSource()
    pieces = source.view()
    low, high = pieces.low - 2.0, pieces.high + 2.0
    picture = slice_picture(pieces, 4, low, high, width=400)
    assert picture.shape[1] == 400 and picture.dtype == np.uint8
    assert np.array_equal(picture, slice_picture(source.view(4), 4, low, high, width=400)), "the layer alone, the same picture"
    black = picture.sum(axis=2) == 0
    assert black[0].all() and black[:, 0].all() and not black[200].all(), "black round the grid, cells inside it"


def test_the_slice_of_the_middle_is_the_slice_at_the_middle_layer():
    pieces = FakeSource().view()
    assert np.array_equal(slice_of_the_middle(pieces, pieces.low, pieces.high),
                          slice_picture(pieces, 4, pieces.low, pieces.high))


def test_a_slice_picture_shows_a_packed_cell_solid_and_the_calm_ones_as_outlines_on_the_black_ground():
    kind, length, packing = uneven_lattice(8, spread=0.0)
    packing = np.ones_like(packing)
    packed, calm = 4 * 64 + 4 * 8 + 4, 2 * 64 + 2 * 8 + 4
    packing[packed] = 50.0
    pieces = cell_pieces(8, kind, length, packing)
    picture = slice_picture(pieces, 4, pieces.low, pieces.high, width=440).astype(int)
    scale = 440 / (pieces.high[0] - pieces.low[0])

    def middle_of(cell):
        middle = pieces.corners[np.argwhere(pieces.cell == cell)[0, 0]].mean(axis=0)
        return picture[picture.shape[0] - int((middle[1] - pieces.low[1]) * scale), int((middle[0] - pieces.low[0]) * scale)]

    assert middle_of(packed).tolist() == list(VOLUME_RED), "the packed volume is drawn solid"
    assert middle_of(calm).sum() == 0, "a volume at the start's mean is clear: the black ground shows"
    assert (picture.sum(axis=2) > 0).sum() > 20 * (picture == VOLUME_RED).all(axis=2).sum() / 10, "and the outlines are there"


def test_saved_frames_step_the_source_and_fit_every_picture_to_the_grids_own_frame(tmp_path):
    source, views = FakeSource(), []

    def picture(pieces, low, high):
        views.append((low.copy(), high.copy(), pieces))
        return slice_of_the_middle(pieces, low, high)

    frames = saved_frames(source, 5, 2, tmp_path, picture)
    assert [frame["step"] for frame in frames] == [0, 2, 4, 5] and source.steps == 5
    for low, high, pieces in views:
        assert (pieces.place[:, 2] == 4).all() and len(pieces.cell) == 64, "the middle layer, laid out alone"
        assert np.allclose(high - low, (pieces.high - pieces.low) * 1.08), "the frame with its margin fills the picture"
    assert (views[-1][1] < views[0][1]).all(), "this source's cells shrink, so its frame does, evenly"
    assert sorted(path.name for path in tmp_path.iterdir()) == [f"step_0000{step}.png" for step in (0, 2, 4, 5)]
    assert frames[0]["bytes"] == (tmp_path / "step_00000.png").stat().st_size and frames[0]["volumes"] == 64
    whole = []
    saved_frames(FakeSource(), 0, 1, tmp_path, lambda pieces, low, high, plane: whole.append((pieces, plane))
                 or slice_of_the_middle(plane, low, high), whole=True)
    assert len(whole[0][0].cell) == 512 and len(whole[0][1].cell) == 64, "the whole grid's blocks and the slice under them"
    with pytest.raises(ValueError):
        saved_frames(source, 5, 0, tmp_path, picture)
