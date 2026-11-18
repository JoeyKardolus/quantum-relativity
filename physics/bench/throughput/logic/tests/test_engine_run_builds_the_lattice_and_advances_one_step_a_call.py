"""Engine run builds the real engine: the lattice's known counts, one engine step per advance."""
import pytest

wp = pytest.importorskip("warp")
if not wp.is_cuda_available():
    pytest.skip("needs CUDA", allow_module_level=True)

from physics.bench.throughput.logic.engine_run import engine_run  # noqa: E402


def test_eight_a_side_is_512_cells_64_volumes_448_joints_and_advance_is_one_step():
    run = engine_run(8)
    start = run.facts()
    assert run.cells == 512
    assert (start["energy_volumes"], start["joints"]) == (64, 448)
    assert (start["steps_done"], start["ids_used"]) == (0, 512)
    assert start["build_seconds"] > 0 and start["bytes_per_cell"] > 0
    run.advance()
    run.advance()
    assert run.facts()["steps_done"] == 2
