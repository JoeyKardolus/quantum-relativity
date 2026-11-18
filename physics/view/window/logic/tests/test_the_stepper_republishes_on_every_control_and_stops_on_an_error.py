"""The stepping thread, the title and the driver variables, on a plain-array source (no engine, no OpenGL)."""
import time

import pytest

from physics.view.drawing.definitions.shown_kinds import SHOW_BOTH, SHOW_ENERGY_VOLUMES
from physics.view.tests.grid_cases import FakeSource
from physics.view.window.definitions.snapshot import Snapshot
from physics.view.window.logic.stepper import Stepper
from physics.view.window.logic.title_line import title_line
from physics.view.window.logic.use_gpu_driver import use_gpu_driver

SIDE = FakeSource.SIDE


def _wait(stepper, wanted, seconds=5.0):
    until = time.time() + seconds
    while time.time() < until:
        snapshot = stepper.latest()
        if snapshot is not None and wanted(snapshot):
            return snapshot
        time.sleep(0.01)
    raise AssertionError("no such snapshot came")


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


def test_the_title_is_one_known_line_and_a_missing_count_shows_as_a_question_mark():
    stats = {"volumes": 64, "joints": 448, "collapses": 2, "tears": 0, "endless": 1, "size": 134335.05,
             "size_growth": 1.25, "amount": 51447814, "warp": 0.99676}
    snapshot = Snapshot(7, None, None, None, 512, None, None, stats, 12.34, True, "", 0, None)
    assert title_line(snapshot, 59.6) == (
        "step 7 | volumes 64 | joints 448 | collapses 2 | tears 0 | endless 1 | size 134,335 (1.2500x) | "
        "quanta 51,447,814 | warp 0.997 | 512 cells drawn | whole | 12.3 steps/s | 60 fps | paused")
    bare = Snapshot(0, None, None, None, 0, None, None, {}, 0.0, False, "ValueError: no", 0, 4)
    assert title_line(bare, 0.0) == (
        "step 0 | volumes ? | joints ? | collapses ? | tears ? | endless ? | size ? | quanta ? | warp ? | "
        "0 cells drawn | slice z 4 | 0.0 steps/s | 0 fps | STOPPED: ValueError: no")


def test_the_driver_variables_are_set_unless_the_user_set_them(monkeypatch):
    monkeypatch.delenv("GALLIUM_DRIVER", raising=False)
    monkeypatch.setenv("MESA_D3D12_DEFAULT_ADAPTER_NAME", "Intel")
    assert use_gpu_driver() == {"GALLIUM_DRIVER": "d3d12", "MESA_D3D12_DEFAULT_ADAPTER_NAME": "Intel"}
