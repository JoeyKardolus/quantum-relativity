"""GPU guard for the cell engine's tests: skip cleanly on a machine without a CUDA device."""
from __future__ import annotations

import pytest
import warp as wp

wp.config.quiet = True
needs_gpu = pytest.mark.skipif(wp.get_cuda_device_count() == 0, reason="the cell engine's tests run on a CUDA GPU")
