"""GPU cases (test helper): skips for tests that need the engine's GPU or a display.

`need_cuda()` skips the calling test unless Warp finds a CUDA device (the engine runs
there). `need_display()` skips it unless pyglet can be loaded and a display is set,
and points Mesa at the GPU first (`use_gpu_driver`).
"""
from __future__ import annotations

import os

import pytest

from physics.view.window.logic.use_gpu_driver import use_gpu_driver


def need_cuda() -> None:
    warp = pytest.importorskip("warp")
    warp.config.quiet = True
    if warp.get_cuda_device_count() == 0:
        pytest.skip("the engine runs on a CUDA GPU")


def need_display() -> None:
    if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        pytest.skip("drawing on the GPU needs a display")
    use_gpu_driver()
    pytest.importorskip("pyglet")
