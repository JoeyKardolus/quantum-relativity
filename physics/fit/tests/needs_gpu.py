"""needs_gpu: a marker that skips a test cleanly on a machine without CUDA (the engine's real runs are on the GPU)."""
import pytest
import warp as wp

wp.config.quiet = True

needs_gpu = pytest.mark.skipif(not wp.is_cuda_available(), reason="needs CUDA: the real runs are made on the GPU")
