"""The window as a whole: a real pyglet window on a plain-array source (needs a display and OpenGL 3.3).

The window shows for a moment. Keys are pressed by calling the window's own handler.
"""
import time

import pytest

from physics.view.drawing.definitions.camera_marks import CORNER, FACING
from physics.view.tests.gpu_cases import need_display
from physics.view.tests.grid_cases import FakeSource


def _until(window, wanted, seconds=5.0):
    until = time.time() + seconds
    while time.time() < until:
        window.switch_to()
        window.dispatch_events()
        window.on_draw()
        if window.snapshot is not None and wanted(window.snapshot):
            return window.snapshot
        time.sleep(0.01)
    raise AssertionError("the window never drew such a snapshot")


def test_the_window_opens_on_the_whole_grid_and_c_switches_to_the_slice():
    need_display()
    from pyglet.window import key
    from physics.view.window.logic.cell_window import CellWindow
    from physics.view.window.logic.stepper import Stepper
    stepper = Stepper(FakeSource(), paused=True)
    try:
        window = CellWindow(stepper, 320, 200)
    except Exception as raised:
        stepper.stop()
        pytest.skip(f"no OpenGL window here: {raised}")
    try:
        first = _until(window, lambda snapshot: True)
        assert first.layer is None and (window.yaw, window.pitch) == CORNER, "the whole grid, from the corner"
        assert window.mesh.count == FakeSource.SIDE ** 3 and window.mesh.pattern_length == 36
        fitted = window.distance
        assert fitted > 1.0, "the view was fitted to the grid"
        window.on_key_press(key.C, 0)
        sliced = _until(window, lambda snapshot: snapshot.layer is not None)
        assert sliced.layer == 4 and (window.yaw, window.pitch) == FACING and window.mesh.count == 64
        window.on_key_press(key.BRACKETRIGHT, 0)
        assert _until(window, lambda snapshot: snapshot.layer == 5).drawn == 64
        window.on_key_press(key._1, 0)
        assert _until(window, lambda snapshot: snapshot.drawn == 0).layer == 5, "an odd layer holds no energy volume"
        window.on_key_press(key.C, 0)
        assert _until(window, lambda snapshot: snapshot.layer is None).drawn == 64 and (window.yaw, window.pitch) == CORNER
        window.on_mouse_scroll(0, 0, 0, 2)
        assert window.distance < fitted
        window.on_key_press(key.F, 0)
        assert window.distance == fitted, "F fits the view again"
        window.on_mouse_drag(0, 0, 10, 100000, 0, 0)
        assert abs(window.pitch) < 1.56, "the view never tips over"
        window.on_key_press(key.Q, 0)
        window.on_key_press(key.SPACE, 0)
        assert _until(window, lambda snapshot: snapshot.step >= 2).paused is False
        window.on_key_press(key.R, 0)
        assert _until(window, lambda snapshot: snapshot.generation == 1).generation == 1
        assert "whole" in window.caption or window.caption == "quantum-relativity"
    finally:
        window.on_close()
    assert not stepper._thread.is_alive(), "closing the window stops the stepper"
