"""Occupied Planck boxes are drawn tinted by their cell's kind, borders brighter, lit against the start.

The fake source here stands in for the engine (physics.potts) in this test only;
the command always runs the real engine.
"""
from __future__ import annotations

import time
from types import SimpleNamespace

import numpy as np
import pytest
import warp as wp

from physics.view import box_points
from physics.view.box_points import BLUE, BORDER_GAIN, ENERGY_VOLUME, GLOW, JOINT, RED, BoxPoints
from physics.view.light_code import LOG_DIMMEST, LOG_SPAN
from physics.view.opacity import AVERAGE_GLOW, PACKED
from physics.view.start_reference import start_reference
from physics.view.stepper import Stepper
from physics.view.title_line import title_line

PACKING_STEPS = 4


def _view(positions, density, kind, border):
    return SimpleNamespace(positions=wp.array(np.asarray(positions, np.float32), device="cpu"),
                           density=wp.array(np.asarray(density, np.float32), device="cpu"),
                           kind=wp.array(np.asarray(kind, np.int32), device="cpu"),
                           border=wp.array(np.asarray(border, np.uint8), device="cpu"))


def _cube(side: int, density: float = 1.0, kind: int = ENERGY_VOLUME, border: int = 0):
    centres = np.stack(np.meshgrid(*[np.arange(side) + 0.5] * 3, indexing="ij"), -1).reshape(-1, 3)
    count = len(centres)
    return _view(centres, np.full(count, density), np.full(count, kind), np.full(count, border))


def _light(colours: np.ndarray) -> np.ndarray:
    codes = colours[:, 3].astype(float)
    return np.where(codes > 0, np.exp(LOG_DIMMEST + LOG_SPAN * (codes - 1.0) / 254.0), 0.0)


def _expected(contrast: float, deep: float) -> float:
    return float(np.exp(np.log(AVERAGE_GLOW / deep) * (1.0 - np.log(contrast) / np.log(PACKED))))


class _PackingSource:
    """A 4^3 block of boxes, its first half energy volume, the rest joint; box 0 doubles."""

    def reset(self) -> None:
        self.density = np.ones(64)
        self.steps = 0

    def __init__(self) -> None:
        self.reset()

    def step(self) -> None:
        if self.steps < PACKING_STEPS:
            self.density[0] *= 2.0
        self.steps += 1

    def view(self):
        centres = np.stack(np.meshgrid(*[np.arange(4) + 10.5] * 3, indexing="ij"), -1).reshape(-1, 3)
        kind = np.where(np.arange(64) < 32, ENERGY_VOLUME, JOINT)
        return _view(centres, self.density, kind, np.zeros(64))

    def stats(self) -> dict:
        return {"total_energy": float(self.density.sum()), "cells": 2, "cell_count_growth": 1.0}


def test_each_occupied_box_is_drawn_at_its_centre_in_its_cell_kind_colour():
    gathered = BoxPoints().boxes(_view([[1.5, 2.5, 3.5], [4.5, 0.5, 5.5]], [4.0, 4.0],
                                       [ENERGY_VOLUME, JOINT], [0, 0]), np.array([1.0, 1.0, 2.0]))
    found = {tuple(position): tuple(colour[:3]) for position, colour
             in zip(gathered.positions.tolist(), gathered.colours.tolist())}
    assert found == {(1.5, 2.5, 3.5): RED, (4.5, 0.5, 5.5): BLUE}
    assert gathered.low.tolist() == [1.5, 0.5, 3.5] and gathered.high.tolist() == [4.5, 2.5, 5.5]


def test_three_vector_positions_are_read_like_rows():
    rows = _view([[1.5, 2.5, 3.5]], [2.0], [JOINT], [1])
    vectors = SimpleNamespace(**{**vars(rows), "positions": wp.array([wp.vec3(1.5, 2.5, 3.5)],
                                                                     dtype=wp.vec3, device="cpu")})
    first, second = (BoxPoints().boxes(view, np.array([1.0, 1.0, 1.0])) for view in (rows, vectors))
    assert np.array_equal(first.positions, second.positions)
    assert np.array_equal(first.colours, second.colours)


def test_an_even_universe_sums_to_the_average_glow_along_a_line_of_sight():
    side = 20
    view = _cube(side, density=0.5)
    reference = start_reference(view.density, view.kind)
    assert np.allclose(reference, [0.5, 1.0, side])
    gathered = BoxPoints().boxes(view, reference)
    assert len(gathered.positions) == side ** 3
    assert abs(_light(gathered.colours).sum() / side ** 2 / AVERAGE_GLOW - 1.0) < 0.05


def test_a_border_box_is_brighter_by_the_border_gain_on_the_same_scale():
    view = _view([[0.5, 0.5, 0.5], [1.5, 0.5, 0.5]], [3.0, 3.0], [JOINT, JOINT], [0, 1])
    inner, border = _light(BoxPoints().boxes(view, np.array([1.0, 1.0, 8.0])).colours)
    assert abs(inner / _expected(3.0, 8.0) - 1.0) < 0.05
    assert abs(border / inner / BORDER_GAIN - 1.0) < 0.1


def test_past_the_cap_boxes_are_thinned_fairly(monkeypatch):
    view = _cube(20, density=4.0)
    full = _light(BoxPoints().boxes(view, np.array([1.0, 1.0, 20.0])).colours).sum()
    monkeypatch.setattr(box_points, "MOST_POINTS", 1000)
    thinned = BoxPoints().boxes(view, np.array([1.0, 1.0, 20.0]))
    assert 850 < len(thinned.positions) <= 1000
    assert abs(_light(thinned.colours).sum() / full - 1.0) < 0.15


def test_negative_density_is_counted_and_drawn_dark():
    gathered = BoxPoints().boxes(_view([[0.5, 0.5, 0.5]], [-1.0], [ENERGY_VOLUME], [0]),
                                 np.array([1.0, 1.0, 1.0]))
    assert gathered.negative == 1 and len(gathered.positions) == 0


@pytest.mark.parametrize("change, message", [({"density": [np.nan]}, "not finite"),
                                              ({"positions": [[np.inf, 0.5, 0.5]]}, "not finite"),
                                              ({"kind": [2]}, "cell kind")])
def test_a_broken_box_stops_the_frame(change, message):
    fields = {"positions": [[0.5, 0.5, 0.5]], "density": [1.0], "kind": [JOINT], "border": [0]}
    with pytest.raises(ValueError, match=message):
        BoxPoints().boxes(_view(**{**fields, **change}), np.array([1.0, 1.0, 1.0]))


def test_arrays_of_different_lengths_are_refused():
    view = _view([[0.5, 0.5, 0.5]], [1.0, 2.0], [JOINT, JOINT], [0, 0])
    with pytest.raises(ValueError, match="positions"):
        BoxPoints().boxes(view, np.array([1.0, 1.0, 1.0]))


def test_a_packing_box_lights_up_against_a_reference_that_never_moves():
    stepper = Stepper(_PackingSource(), steps_per_frame=1, paused=True, look=GLOW)
    try:
        start = _wait(stepper, lambda snapshot: True)
        assert start.step == 0 and start.low.tolist() == [10.5] * 3 and start.high.tolist() == [13.5] * 3
        stepper.toggle_pause()
        later = _wait(stepper, lambda snapshot: snapshot.step >= PACKING_STEPS)
        packed = [index for index, position in enumerate(later.positions.tolist())
                  if tuple(position) == (10.5, 10.5, 10.5)]
        assert len(packed) == 1 and tuple(later.colours[packed[0]][:3]) == RED
        assert abs(_light(later.colours[packed])[0] / _expected(2.0 ** PACKING_STEPS, 4.0) - 1.0) < 0.05
        line = title_line(later, 60.0)
        assert "cells 2 (1.00x)" in line and "joints ?" in line and "64 boxes drawn" in line
    finally:
        stepper.stop()


def _wait(stepper, wanted, seconds: float = 10.0):
    ends = time.perf_counter() + seconds
    while time.perf_counter() < ends:
        snapshot = stepper.latest()
        if snapshot is not None and wanted(snapshot):
            return snapshot
        time.sleep(0.01)
    raise AssertionError(f"no snapshot in {seconds} s; error: {stepper.error!r}")
