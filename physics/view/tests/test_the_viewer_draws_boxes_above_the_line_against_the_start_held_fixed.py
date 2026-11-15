"""The viewer gathers boxes above the shown line, coloured against the start's mean held fixed.

The fake source here stands in for physics.fluid in this test only; the command
always runs the real engine.
"""
from __future__ import annotations

import time

import numpy as np
import pytest
import warp as wp

from physics.view.box_points import BLUE, RED, SHOWN, BoxPoints
from physics.view.start_reference import start_reference
from physics.view.stepper import Stepper
from physics.view.title_line import title_line

SIDE = 6


class _PackingSource:
    """A ball of A and B energy that doubles its middle box's energy each step."""

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


def test_only_boxes_above_the_line_are_drawn_at_their_centres():
    pulling = np.zeros((SIDE, SIDE, SIDE), np.float32)
    pushing = np.zeros((SIDE, SIDE, SIDE), np.float32)
    pulling[1, 2, 3] = 1.0
    pushing[4, 0, 5] = 1.0
    pulling[5, 5, 5] = SHOWN * 0.5
    gathered = BoxPoints().boxes(wp.array(pulling, device="cpu"), wp.array(pushing, device="cpu"),
                                 np.array([1.0, 1.0]))
    found = {tuple(position): tuple(colour[:3]) for position, colour
             in zip(gathered.positions.tolist(), gathered.colours.tolist())}
    assert found == {(1.5, 2.5, 3.5): RED, (4.5, 0.5, 5.5): BLUE}


def test_a_box_that_is_not_finite_stops_the_frame():
    pulling = np.ones((SIDE, SIDE, SIDE), np.float32)
    pulling[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="not finite"):
        BoxPoints().boxes(wp.array(pulling, device="cpu"),
                          wp.array(np.ones_like(pulling), device="cpu"), np.ones(2))


def test_the_reference_is_the_start_mean_of_each_kind_over_occupied_boxes():
    source = _PackingSource()
    assert np.allclose(start_reference(source.a_energy, source.b_energy), [1.0, 0.5])


def test_a_packing_box_grows_solid_against_a_reference_that_never_moves():
    stepper = Stepper(_PackingSource(), steps_per_frame=1, paused=True)
    try:
        start = _wait(stepper, lambda snapshot: True)
        assert len(start.positions) == 8 and start.step == 0
        assert np.all(start.colours[:, 3] == start.colours[0, 3])
        stepper.toggle_pause()
        later = _wait(stepper, lambda snapshot: snapshot.step >= 5)
        alpha = later.colours[:, 3].astype(int)
        assert alpha.max() == 255 and alpha.max() > start.colours[0, 3]
        assert sorted(alpha)[:7] == [start.colours[0, 3]] * 7
        line = title_line(later, 60.0)
        assert "lumps 1" in line and "sealed ?" in line and "8 boxes drawn" in line
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
