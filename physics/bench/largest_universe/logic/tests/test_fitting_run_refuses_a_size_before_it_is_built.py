"""Fitting run refuses a size before building it, and builds the real engine when it fits."""
import pytest

wp = pytest.importorskip("warp")
if not wp.is_cuda_available():
    pytest.skip("needs CUDA", allow_module_level=True)

from physics.bench.largest_universe.definitions.memory_law import bytes_needed  # noqa: E402
from physics.bench.largest_universe.logic.fitting_run import fitting_run  # noqa: E402


def test_one_byte_short_is_refused_with_nothing_allocated_and_exactly_enough_is_built():
    device = wp.get_device("cuda:0")
    held = wp.get_mempool_used_mem_current(device)
    with pytest.raises(MemoryError):
        fitting_run(8, free_bytes=bytes_needed(8) - 1)
    assert wp.get_mempool_used_mem_current(device) == held
    run = fitting_run(8, free_bytes=bytes_needed(8))
    assert run.cells == 512 and run.facts()["steps_done"] == 0
