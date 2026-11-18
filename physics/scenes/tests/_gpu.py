"""GPU guard and the settings the scene tests run on.

`needs_gpu` skips a test cleanly on a machine without a CUDA device. `HELD` is the
engine's own "held" test run (physics/engine/tests/conftest.py): the defaults with
`push_strength = 40` and `tear_length = 200`, on which the grid stands (no collapse, no
tear) for the few steps a test takes. The default universe falls into a few volumes
within 40 steps, which is the engine's finding and not what a scene's test is about.
"""
from __future__ import annotations

import pytest
import warp as wp

from physics.engine import Settings

wp.config.quiet = True
needs_gpu = pytest.mark.skipif(wp.get_cuda_device_count() == 0, reason="the scenes run the engine on a CUDA GPU")
HELD = Settings(push_strength=40.0, tear_length=200.0)
