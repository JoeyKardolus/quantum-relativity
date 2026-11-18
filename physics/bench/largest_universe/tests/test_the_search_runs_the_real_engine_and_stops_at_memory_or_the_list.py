"""The whole search on the real engine: a rate every size meets, and a GPU with nothing free."""
import pytest

wp = pytest.importorskip("warp")
if not wp.is_cuda_available():
    pytest.skip("needs CUDA", allow_module_level=True)

from physics.bench.largest_universe import fitting_run, largest_universe  # noqa: E402
from physics.bench.largest_universe.definitions.limits import MEMORY, SIZE_LIST  # noqa: E402


def test_at_a_rate_every_size_meets_the_largest_of_the_list_is_found_with_its_512_cells():
    search = largest_universe(fitting_run, [4, 8], min_rate=1e-6, steps=2, warmup=1)
    assert (search.size, search.cells, search.limit) == (8, 512, SIZE_LIST)
    assert [(probe.size, probe.cells) for probe in search.probes] == [(4, 64), (8, 512)]
    assert all(probe.facts["steps_done"] == 3 for probe in search.probes)
    assert (search.probes[1].facts["energy_volumes"], search.probes[1].facts["joints"]) == (64, 448)


def test_with_nothing_free_the_smallest_is_refused_and_memory_is_named():
    search = largest_universe(lambda size: fitting_run(size, free_bytes=0), [4, 8])
    assert (search.size, search.limit) == (None, MEMORY)
    assert [(probe.size, probe.fits) for probe in search.probes] == [(4, False)]
