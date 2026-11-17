"""The slice draws one layer of cells, the whole grid its blocks, far cells first; kinds can be shown alone; the stepper republishes while paused.

A fake source on plain arrays (no engine, no OpenGL): an 8-place lattice whose cells shrink a little each step.
"""
import time

import numpy as np
import pytest

from physics.view import (cell_pieces, far_to_near, grid_triangles, middle_layer, orbit_camera, saved_frames,
                          slice_of_the_middle, slice_picture, title_line, write_png)
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
    corners, colours, pattern, drawn = grid_triangles(pieces, SHOW_BOTH, 4)
    assert drawn == SIDE * SIDE and corners.shape == (drawn, 4, 3) and colours.shape == (drawn, 4)
    assert corners.dtype == np.float32 and colours.dtype == np.uint8
    assert pattern.tolist() == [0, 1, 2, 0, 2, 3], "two triangles a cell"
    assert np.ptp(corners[:, :, 2], axis=1).max() < 3.0, "each cell's face is flat, in the middle of its block"
    _, reds, _, volumes = grid_triangles(pieces, SHOW_ENERGY_VOLUMES, 4)
    _, blues, _, joints = grid_triangles(pieces, SHOW_JOINTS, 4)
    assert volumes == 16 and joints == 48 and (reds[:, 0] > reds[:, 2]).all() and (blues[:, 2] > blues[:, 0]).all()
    assert grid_triangles(pieces, SHOW_ENERGY_VOLUMES, 5)[3] == 0, "an odd layer holds only joints"
    with pytest.raises(ValueError):
        grid_triangles(pieces, 7, 4)


def test_the_whole_grid_draws_every_cell_as_a_block_so_the_inside_shows_through():
    pieces = FakeSource().view()
    corners, colours, pattern, drawn = grid_triangles(pieces, SHOW_BOTH, None)
    assert drawn == SIDE ** 3 and corners.shape == (drawn, 8, 3) and colours.shape == (drawn, 4)
    assert len(pattern) == 36 and sorted(set(pattern.tolist())) == list(range(8)), "six faces, twelve triangles"
    assert np.array_equal(corners, pieces.corners)
    assert grid_triangles(pieces, SHOW_ENERGY_VOLUMES, None)[3] == (SIDE // 2) ** 3
    assert grid_triangles(pieces, SHOW_JOINTS, None)[3] == SIDE ** 3 - (SIDE // 2) ** 3


def test_cells_are_ordered_far_to_near_from_wherever_the_camera_stands():
    centres = FakeSource().view().corners.mean(axis=1)
    middle = centres.mean(axis=0)
    for yaw, pitch in ((0.0, 0.0), (0.6 + np.pi, -0.35), (2.0, 1.2)):
        eye = middle + 400.0 * np.array([np.cos(pitch) * np.sin(yaw), np.sin(pitch), np.cos(pitch) * np.cos(yaw)])
        order = far_to_near(centres, orbit_camera(middle, 400.0, yaw, pitch, 1.5))
        assert order.dtype == np.uint32 and sorted(order.tolist()) == list(range(len(centres))), "every cell once"
        along = (centres[order] - eye) @ (middle - eye) / 400.0
        assert along[0] == along.max() and along[-1] == along.min()
        assert np.diff(along).max() < np.ptp(along) / 1000.0, "never a nearer cell before a farther one"
    assert len(far_to_near(np.zeros((0, 3)), orbit_camera(middle, 400.0, 0.0, 0.0, 1.5))) == 0


def test_the_stepper_starts_on_the_whole_grid_and_every_control_republishes_while_paused():
    stepper = Stepper(FakeSource(), paused=True)
    try:
        start = _wait(stepper, lambda snapshot: True)
        assert start.layer is None and start.drawn == SIDE ** 3 and start.corners.shape == (SIDE ** 3, 8, 3)
        assert "whole" in title_line(start, 60.0) and "paused" in title_line(start, 60.0)
        stepper.move_slice(1)
        stepper.toggle_whole()
        first = _wait(stepper, lambda snapshot: snapshot.layer is not None)
        assert first.layer == 4 and first.step == 0 and first.paused and first.drawn == 64, "C goes to the middle slice"
        assert "slice z 4" in title_line(first, 60.0) and len(first.pattern) == 6
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
    red = (picture[..., 0] > picture[..., 2]) & (picture[..., 0] > 10)
    blue = (picture[..., 2] > picture[..., 0]) & (picture[..., 2] > 10)
    black = picture.sum(axis=2) == 0
    assert red.sum() + blue.sum() + black.sum() == picture[..., 0].size, "only reds, blues and the black round the grid"
    assert 0.6 < red.sum() / (red.sum() + blue.sum()) < 0.95, "bricks with thin mortar"
    assert black[0].all() and black[:, 0].all() and not black[200, 40:360].any(), "no gap inside the grid"
    path = tmp_path / "slice.png"
    assert write_png(path, picture) == path.stat().st_size
    assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n" and path.read_bytes()[16:24] == bytes(
        [0, 0, 1, 144, *picture.shape[0].to_bytes(4, "big")])


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
