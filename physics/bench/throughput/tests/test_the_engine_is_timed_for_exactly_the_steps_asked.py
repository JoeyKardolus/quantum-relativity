"""Throughput as a whole, on the real engine: the steps asked for are the steps the universe took."""
import pytest

wp = pytest.importorskip("warp")
if not wp.is_cuda_available():
    pytest.skip("needs CUDA", allow_module_level=True)

from physics.bench.throughput import engine_run, throughput  # noqa: E402


def test_three_timed_steps_after_one_warmup_leave_the_universe_at_step_four():
    run = engine_run(8)
    rates = throughput(run.advance, run.cells, steps=3, warmup=1)
    assert run.facts()["steps_done"] == 4
    assert (rates.cells, rates.steps) == (512, 3)
    assert rates.seconds > 0
    assert rates.cell_steps_per_second == pytest.approx(512 * 3 / rates.seconds)
