"""The viewer draws every dense box plus a fair sample of the rest, lit against the start held fixed.

The fake source here stands in for physics.fluid in this test only; the command
always runs the real engine.
"""
from __future__ import annotations

import time

import numpy as np
import pytest
import warp as wp

from physics.view import box_points
from physics.view.box_points import BLUE, RED, SAMPLE, BoxPoints
from physics.view.light_code import LOG_DIMMEST, LOG_SPAN
from physics.view.opacity import AVERAGE_GLOW, PACKED
from physics.view.start_reference import start_reference
from physics.view.stepper import Stepper
from physics.view.title_line import title_line

SIDE = 6
PACKING_STEPS = 4


class _PackingSource:
    """A ball of A and B energy whose middle box doubles its A for PACKING_STEPS steps."""

    side = SIDE

    def reset(self) -> None:
        self.a = np.zeros((SIDE, SIDE, SIDE), np.float64)
        self.b = np.zeros((SIDE, SIDE, SIDE), np.float64)
        self.a[2:4, 2:4, 2:4] = 1.0
        self.b[2:4, 2:4, 2:4] = 0.5
        self.steps = 0

    def __init__(self) -> None:
        self.reset()

    def step(self) -> None:
        if self.steps < PACKING_STEPS:
            self.a[2, 2, 2] *= 2.0
        self.steps += 1

    @property
    def a_energy(self):
        return wp.array(self.a, device="cpu")

    @property
    def b_energy(self):
        return wp.array(self.b, device="cpu")

    def stats(self) -> dict:
        return {"total_energy": float(self.a.sum() + self.b.sum()), "lumps": 1}


def _light(colours: np.ndarray) -> np.ndarray:
    codes = colours[:, 3].astype(float)
    return np.where(codes > 0, np.exp(LOG_DIMMEST + LOG_SPAN * (codes - 1.0) / 254.0), 0.0)


def _boxes(pulling: np.ndarray, pushing: np.ndarray, reference=(1.0, 1.0)):
    return BoxPoints().boxes(wp.array(pulling, device="cpu"), wp.array(pushing, device="cpu"),
                             np.asarray(reference, dtype=float))


def test_dense_boxes_are_drawn_at_their_centres_in_their_kind_colour():
    pulling = np.zeros((SIDE, SIDE, SIDE), np.float32)
    pushing = np.zeros((SIDE, SIDE, SIDE), np.float32)
    pulling[1, 2, 3] = 4.0
    pushing[4, 0, 5] = 4.0
    gathered = _boxes(pulling, pushing)
    found = {tuple(position): tuple(colour[:3]) for position, colour
             in zip(gathered.positions.tolist(), gathered.colours.tolist())}
    assert found == {(1.5, 2.5, 3.5): RED, (4.5, 0.5, 5.5): BLUE} and gathered.dense == 2


def test_an_even_universe_sums_to_the_average_glow_along_a_line_of_sight():
    side = 40
    even = np.full((side, side, side), 0.5, np.float32)
    gathered = _boxes(even, even, (0.5, 0.5))
    assert gathered.dense == 0
    assert abs(len(gathered.positions) - SAMPLE * side ** 3) < 5 * np.sqrt(SAMPLE * side ** 3)
    per_line_of_sight = _light(gathered.colours).sum() / side ** 2
    assert abs(per_line_of_sight / AVERAGE_GLOW - 1.0) < 0.2


def test_past_the_cap_dense_boxes_are_thinned_fairly(monkeypatch):
    side = 20
    dense = np.full((side, side, side), 4.0, np.float32)
    full = _light(_boxes(dense, np.zeros_like(dense)).colours).sum()
    monkeypatch.setattr(box_points, "MOST_POINTS", 1000)
    thinned = _boxes(dense, np.zeros_like(dense))
    assert thinned.dense == side ** 3 and len(thinned.positions) <= 1000
    assert len(thinned.positions) > 850
    assert abs(_light(thinned.colours).sum() / full - 1.0) < 0.15


def test_a_box_that_is_not_finite_stops_the_frame():
    pulling = np.ones((SIDE, SIDE, SIDE), np.float32)
    pulling[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="not finite"):
        _boxes(pulling, np.ones_like(pulling))


def test_the_reference_is_the_start_mean_of_each_kind_over_occupied_boxes():
    source = _PackingSource()
    assert np.allclose(start_reference(source.a_energy, source.b_energy), [1.0, 0.5])


def test_a_packing_box_lights_up_against_a_reference_that_never_moves():
    stepper = Stepper(_PackingSource(), steps_per_frame=1, paused=True)
    try:
        start = _wait(stepper, lambda snapshot: True)
        assert start.step == 0
        assert not any(tuple(position) == (2.5, 2.5, 2.5) for position in start.positions.tolist())
        stepper.toggle_pause()
        later = _wait(stepper, lambda snapshot: snapshot.step >= PACKING_STEPS)
        middle = [index for index, position in enumerate(later.positions.tolist())
                  if tuple(position) == (2.5, 2.5, 2.5)]
        assert len(middle) == 1
        contrast = (2.0 ** PACKING_STEPS + 0.5) / 1.5
        expected = np.exp(np.log(AVERAGE_GLOW / SIDE) * (1.0 - np.log(contrast) / np.log(PACKED)))
        assert abs(_light(later.colours[middle])[0] / expected - 1.0) < 0.05
        line = title_line(later, 60.0)
        assert "lumps 1" in line and "sealed ?" in line and "boxes drawn" in line
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
