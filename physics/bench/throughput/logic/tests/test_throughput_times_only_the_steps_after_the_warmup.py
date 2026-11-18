"""Throughput calls advance warmup + steps times and times only the steps: a fake that sleeps a known time."""
import time

import pytest

from physics.bench.throughput.logic.throughput import throughput


def test_advance_is_called_warmup_plus_steps_times():
    calls = []
    rates = throughput(lambda: calls.append(1), cells=1000, steps=7, warmup=2)
    assert len(calls) == 9
    assert (rates.cells, rates.steps) == (1000, 7)
    assert rates.cell_steps_per_second == pytest.approx(1000 * rates.steps_per_second)
    assert rates.steps_per_second == pytest.approx(7 / rates.seconds)


def test_a_slow_warmup_is_not_in_the_time_and_every_timed_step_is():
    calls = []

    def advance():
        calls.append(1)
        time.sleep(0.5 if len(calls) == 1 else 0.05)

    rates = throughput(advance, cells=1, steps=4, warmup=1)
    assert 0.2 <= rates.seconds < 0.5
    assert rates.steps_per_second <= 20.0


def test_a_refused_plan_never_calls_advance():
    calls = []
    with pytest.raises(ValueError):
        throughput(lambda: calls.append(1), cells=1, steps=0, warmup=3)
    assert calls == []
