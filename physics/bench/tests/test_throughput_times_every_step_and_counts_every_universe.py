"""Throughput calls advance warmup + steps times and scales the rate by universes and boxes.

A host-only advance that counts its calls; no GPU needed.
"""
import pytest

from physics.bench import throughput


def test_calls_and_rates():
    calls = []
    result = throughput(lambda: calls.append(1), universe_count=8, boxes_per_universe=1000, steps=7, warmup=2)
    assert len(calls) == 9
    assert result["steps"] == 7
    assert result["universe_steps_per_second"] == pytest.approx(8 * result["steps_per_second"])
    assert result["box_steps_per_second"] == pytest.approx(8000 * result["steps_per_second"])
    with pytest.raises(ValueError):
        throughput(lambda: None, 1, 1, steps=0)
