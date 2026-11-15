"""GPU guard for the scene tests: skip cleanly on a machine without a CUDA device."""
from __future__ import annotations

import pytest
import warp as wp

wp.config.quiet = True
needs_gpu = pytest.mark.skipif(wp.get_cuda_device_count() == 0, reason="the scenes run the cell engine on a CUDA GPU")
