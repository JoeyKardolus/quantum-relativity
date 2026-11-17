"""The slice draws one layer of cells, the whole grid its blocks; kinds can be shown alone; the stepper republishes while paused.

A fake source on plain arrays (no engine, no OpenGL): an 8-place lattice whose cells shrink a little each step.
"""
import time

import numpy as np
import pytest

from physics.view import cell_pieces, grid_triangles, middle_layer, slice_picture, title_line, write_png
from physics.view.grid_triangles import SHOW_BOTH, SHOW_ENERGY_VOLUMES, SHOW_JOINTS
from physics.view.stepper import Stepper
from physics.view.tests.grid_cases import uneven_lattice

SIDE = 8


class FakeSource:
    def __init__(self, fail_at: int = -1):
        self.fail_at = fail_at
        self.reset()

    def reset(self):
        self.kind, self.length, self.packing = uneven_lattice(SIDE, spread=0.1)
        self.steps = 0

    def step(self):
        if self.steps == self.fail_at:
            raise RuntimeError("the source broke")
        self.length = self.length * 0.99
        self.steps += 1

    def view(self):
        return cell_pieces(SIDE, self.kind, self.length, self.packing)

    def stats(self):
        return {"volumes": int((self.kind == 1).sum()), "joints": int((self.kind == -1).sum()), "size": 1.0}


def _wait(stepper, wanted, seconds=5.0):
    until = time.time() + seconds
    while time.time() < until:
        snapshot = stepper.latest()
        if snapshot is not None and wanted(snapshot):
            return snapshot
        time.sleep(0.01)
    raise AssertionError("no such snapshot came")


def test_a_slice_is_one_layer_two_triangles_a_cell_and_kinds_can_be_shown_alone():
    pieces = FakeSource().view()
    assert middle_layer(SIDE) == 4 and middle_layer(6) == 2 and middle_layer(32) == 16
    positions, colours, drawn = grid_triangles(pieces, SHOW_BOTH, 4)
    assert drawn == SIDE * SIDE and positions.shape == (drawn * 6, 3) and colours.shape == (drawn * 6, 4)
    assert positions.dtype == np.float32 and colours.dtype == np.uint8
    flat = positions.reshape(drawn, 6, 3)
    assert np.ptp(flat[:, :, 2], axis=1).max() < 3.0, "each cell's face is flat, in the middle of its block"
    _, reds, volumes = grid_triangles(pieces, SHOW_ENERGY_VOLUMES, 4)
    _, blues, joints = grid_triangles(pieces, SHOW_JOINTS, 4)
    assert volumes == 16 and joints == 48 and (reds[:, 0] > reds[:, 2]).all() and (blues[:, 2] > blues[:, 0]).all()
    assert grid_triangles(pieces, SHOW_ENERGY_VOLUMES, 5)[2] == 0, "an odd layer holds only joints"
    with pytest.raises(ValueError):
        grid_triangles(pieces, 7, 4)


def test_the_whole_grid_draws_blocks_and_with_both_kinds_only_the_outside():
    pieces = FakeSource().view()
    positions, _, drawn = grid_triangles(pieces, SHOW_BOTH, None)
    assert drawn == SIDE ** 3 - (SIDE - 2) ** 3 and len(positions) == drawn * 36
    assert grid_triangles(pieces, SHOW_ENERGY_VOLUMES, None)[2] == (SIDE // 2) ** 3
    assert grid_triangles(pieces, SHOW_JOINTS, None)[2] == SIDE ** 3 - (SIDE // 2) ** 3


def test_the_stepper_starts_on_the_middle_slice_and_every_control_republishes_while_paused():
    stepper = Stepper(FakeSource(), paused=True)
    try:
        first = _wait(stepper, lambda snapshot: True)
        assert first.layer == 4 and first.step == 0 and first.paused and first.drawn == 64
        assert "slice z 4" in title_line(first, 60.0) and "paused" in title_line(first, 60.0)
        stepper.move_slice(1)
        assert _wait(stepper, lambda snapshot: snapshot.layer == 5).drawn == 64
        stepper.move_slice(-6)
        assert _wait(stepper, lambda snapshot: snapshot.layer == 7).step == 0, "the slice goes round the wrapping grid"
        stepper.show(SHOW_ENERGY_VOLUMES)
        assert _wait(stepper, lambda snapshot: snapshot.drawn == 0).layer == 7
        stepper.toggle_whole()
        whole = _wait(stepper, lambda snapshot: snapshot.layer is None)
        assert whole.drawn == 64 and "whole" in title_line(whole, 60.0)
        stepper.move_slice(3)
        stepper.toggle_whole()
        assert _wait(stepper, lambda snapshot: snapshot.layer is not None).layer == 7
        with pytest.raises(ValueError):
            stepper.show(0)
        stepper.show(SHOW_BOTH)
        stepper.toggle_pause()
        running = _wait(stepper, lambda snapshot: snapshot.step >= 3)
        assert not running.paused and running.drawn == 64
        assert (running.high - running.low < first.high - first.low).all(), "the drawing follows the cells' sizes"
        stepper.reset()
        assert _wait(stepper, lambda snapshot: snapshot.generation == 1).generation == 1
    finally:
        stepper.stop()


def test_a_source_that_raises_stops_stepping_and_shows_the_error():
    stepper = Stepper(FakeSource(fail_at=2))
    try:
        stopped = _wait(stepper, lambda snapshot: snapshot.error)
        assert "the source broke" in stopped.error and stopped.paused and stopped.step == 2
        assert "STOPPED" in title_line(stopped, 1.0)
    finally:
        stepper.stop()


def test_a_slice_picture_is_the_same_cells_in_the_same_places_and_saves_as_a_png(tmp_path):
    source = FakeSource()
    pieces = source.view()
    low, high = pieces.low - 2.0, pieces.high + 2.0
    picture = slice_picture(pieces, 4, low, high, width=400)
    assert picture.shape[1] == 400 and picture.dtype == np.uint8
    assert np.array_equal(picture, slice_picture(source.view(), 4, low, high, width=400)), "the same grid, the same picture"
    red = (picture[..., 0] > picture[..., 2]) & (picture[..., 0] > 60)
    blue = (picture[..., 2] > picture[..., 0]) & (picture[..., 2] > 60)
    black = picture.sum(axis=2) == 0
    assert red.sum() + blue.sum() + black.sum() == picture[..., 0].size, "only reds, blues and the black round the grid"
    assert 0.6 < red.sum() / (red.sum() + blue.sum()) < 0.95, "bricks with thin mortar"
    assert black[0].all() and black[:, 0].all() and not black[200, 40:360].any(), "no gap inside the grid"
    path = tmp_path / "slice.png"
    assert write_png(path, picture) == path.stat().st_size
    assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n" and path.read_bytes()[16:24] == bytes(
        [0, 0, 1, 144, *picture.shape[0].to_bytes(4, "big")])
