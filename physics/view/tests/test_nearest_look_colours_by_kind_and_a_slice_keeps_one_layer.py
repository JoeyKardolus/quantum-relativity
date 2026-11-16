"""The nearest look colours every shown box by kind with density as a tint; a slice keeps one box layer; the view refits as space grows.

The fake sources here stand in for the engine (physics.engine) in this test only;
the command always runs the real engine.
"""
from __future__ import annotations

import math
import time
from types import SimpleNamespace

import numpy as np
import pytest
import warp as wp

from physics.view.box_points import (BLUE, ENERGY_VOLUME, GLOW, JOINT, NEAREST, NEAREST_DIMMEST, RED,
                                     SHOW_JOINTS, BoxPoints)
from physics.view.opacity import PACKED
from physics.view.start_reference import start_reference
from physics.view.stepper import Stepper
from physics.view.title_line import title_line
from physics.view.view_fit import REFIT_GROWTH, view_fit


def _view(positions, density, kind, border):
    return SimpleNamespace(positions=wp.array(np.asarray(positions, np.float32), device="cpu"),
                           density=wp.array(np.asarray(density, np.float32), device="cpu"),
                           kind=wp.array(np.asarray(kind, np.int32), device="cpu"),
                           border=wp.array(np.asarray(border, np.uint8), device="cpu"))


def _checkerboard(side: int = 4):
    """An even cube of boxes, alternating energy volume and joint, at density 1."""
    centres = np.stack(np.meshgrid(*[np.arange(side) + 0.5] * 3, indexing="ij"), -1).reshape(-1, 3)
    kind = np.where(np.floor(centres).astype(int).sum(1) % 2 == 0, ENERGY_VOLUME, JOINT)
    return _view(centres, np.ones(len(centres)), kind, np.zeros(len(centres)))


def _shaded(tint, shade: float) -> tuple:
    return tuple(int(round(channel * shade)) for channel in tint)


@pytest.mark.parametrize("kind, tint", [(ENERGY_VOLUME, RED), (JOINT, BLUE)])
def test_nearest_draws_every_box_in_its_kind_colour_with_density_only_as_a_fixed_tint(kind, tint):
    contrasts = [0.0, 1.0 / PACKED, 1.0, PACKED, 1000.0]
    count = len(contrasts)
    positions = np.column_stack([np.arange(count) + 0.5, np.full(count, 0.5), np.full(count, 0.5)])
    view = _view(positions, np.array(contrasts) * 2.0, np.full(count, kind), np.ones(count))
    gathered = BoxPoints().boxes(view, np.array([2.0, 2.0, 30.0]), look=NEAREST)
    order = np.argsort(gathered.positions[:, 0])
    colours = gathered.colours[order]
    assert len(colours) == count and set(colours[:, 3].tolist()) == {255}
    middle = NEAREST_DIMMEST + (1.0 - NEAREST_DIMMEST) * 0.5
    expected = [_shaded(tint, shade) for shade in (NEAREST_DIMMEST, NEAREST_DIMMEST, middle, 1.0, 1.0)]
    assert [tuple(colour[:3]) for colour in colours.tolist()] == expected


def test_nearest_shows_an_even_start_as_a_full_checkerboard_where_glow_is_near_black():
    view = _checkerboard()
    reference = start_reference(view.density, view.kind)
    nearest = BoxPoints().boxes(view, reference, look=NEAREST)
    glow = BoxPoints().boxes(view, reference, look=GLOW)
    assert len(nearest.positions) == 64
    tints = {tuple(position): tuple(colour[:3]) for position, colour
             in zip(nearest.positions.tolist(), nearest.colours.tolist())}
    middle = NEAREST_DIMMEST + (1.0 - NEAREST_DIMMEST) * 0.5
    assert tints[(0.5, 0.5, 0.5)] == _shaded(RED, middle) and tints[(1.5, 0.5, 0.5)] == _shaded(BLUE, middle)
    assert glow.colours[:, 3].max() < nearest.colours[:, 3].min()


def test_an_unknown_look_is_refused():
    with pytest.raises(ValueError, match="look"):
        BoxPoints().boxes(_checkerboard(), np.array([1.0, 1.0, 4.0]), look=7)


@pytest.mark.parametrize("offset, centre, used", [(0, 2.5, 0), (-2, 0.5, -2), (1, 3.5, 1), (5, 3.5, 1), (-9, 0.5, -2)])
def test_a_slice_keeps_one_box_layer_moved_from_the_middle_and_held_inside(offset, centre, used):
    gathered = BoxPoints().boxes(_checkerboard(), np.array([1.0, 1.0, 4.0]), look=NEAREST, slice_offset=offset)
    assert len(gathered.positions) == 16
    assert set(gathered.positions[:, 2].tolist()) == {centre}
    assert gathered.slice_at == centre and gathered.slice_offset == used
    assert gathered.low.tolist() == [0.5] * 3 and gathered.high.tolist() == [3.5] * 3


def test_without_a_slice_every_layer_is_drawn_and_no_slice_is_reported():
    gathered = BoxPoints().boxes(_checkerboard(), np.array([1.0, 1.0, 4.0]), look=NEAREST)
    assert len(gathered.positions) == 64 and math.isnan(gathered.slice_at) and gathered.slice_offset is None


def test_a_slice_and_a_kind_choice_combine():
    gathered = BoxPoints().boxes(_checkerboard(), np.array([1.0, 1.0, 4.0]), SHOW_JOINTS, NEAREST, 0)
    assert len(gathered.positions) == 8 and {tuple(colour[:3]) for colour in gathered.colours.tolist()} == {
        _shaded(BLUE, NEAREST_DIMMEST + (1.0 - NEAREST_DIMMEST) * 0.5)}


class _EvenSource:
    """A 4^3 checkerboard of boxes that never changes."""

    steps = 0

    def reset(self) -> None:
        pass

    def step(self) -> None:
        self.steps += 1

    def view(self):
        return _checkerboard()

    def stats(self) -> dict:
        return {}


def test_the_stepper_starts_nearest_and_switches_look_and_slice_while_paused():
    stepper = Stepper(_EvenSource(), paused=True)
    try:
        start = _wait(stepper, lambda snapshot: True)
        assert start.look == NEAREST and set(start.colours[:, 3].tolist()) == {255}
        assert "nearest" in title_line(start, 60.0) and "slice" not in title_line(start, 60.0)
        stepper.move_slice(1)
        assert stepper.slice_offset == 0
        stepper.draw_as(GLOW)
        glow = _wait(stepper, lambda snapshot: snapshot.look == GLOW)
        assert "glow" in title_line(glow, 60.0)
        stepper.toggle_slice()
        middle = _wait(stepper, lambda snapshot: len(snapshot.positions) == 16)
        assert middle.slice_at == 2.5 and "slice z 2.5" in title_line(middle, 60.0)
        for _ in range(5):
            stepper.move_slice(1)
        _wait(stepper, lambda snapshot: snapshot.slice_at == 3.5)
        _wait(stepper, lambda snapshot: stepper.slice_offset == 1, settle=True)
        stepper.move_slice(-1)
        _wait(stepper, lambda snapshot: snapshot.slice_at == 2.5)
        stepper.toggle_slice()
        _wait(stepper, lambda snapshot: len(snapshot.positions) == 64 and math.isnan(snapshot.slice_at))
        with pytest.raises(ValueError, match="look"):
            stepper.draw_as(5)
    finally:
        stepper.stop()


def test_the_view_fits_first_then_only_when_space_outgrows_the_fit():
    centre, distance, extent = view_fit([0.5, 0.5, 0.5], [10.5, 4.5, 2.5], 45.0)
    assert np.allclose(centre, [5.5, 2.5, 1.5]) and extent == 10.0
    assert distance == pytest.approx(1.1 * 10.0 / np.sin(np.radians(22.5)))
    grown = 10.0 * REFIT_GROWTH
    assert view_fit([0.0] * 3, [grown] * 3, 45.0, fitted_extent=10.0) is None
    assert view_fit([0.0] * 3, [grown + 0.5] * 3, 45.0, fitted_extent=10.0)[2] == grown + 0.5
    assert view_fit([0.0] * 3, [2.0] * 3, 45.0, fitted_extent=10.0) is None
    assert view_fit([0.0] * 3, [0.0] * 3, 45.0)[2] == 1.0


def _wait(stepper, wanted, seconds: float = 10.0, settle: bool = False):
    ends = time.perf_counter() + seconds
    while time.perf_counter() < ends:
        snapshot = stepper.latest()
        if (snapshot is not None or settle) and wanted(snapshot):
            return snapshot
        time.sleep(0.01)
    raise AssertionError(f"no snapshot in {seconds} s; error: {stepper.error!r}")
