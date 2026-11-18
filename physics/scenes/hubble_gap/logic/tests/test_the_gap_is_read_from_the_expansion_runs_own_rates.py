"""The early and late rates are the outer quarters of the expansion run the scene carries, and their ratio and gap follow."""
import numpy as np
import pytest

from physics.scenes.hubble_gap.logic.hubble_gap import hubble_gap
from physics.scenes.tests._gpu import HELD, needs_gpu


@needs_gpu
def test_eight_rate_samples_give_two_early_and_two_late():
    found = hubble_gap(8, steps=8, every=1, settings=HELD)
    rates = np.array(found["expansion"]["size_rate"])
    assert rates.size == 8
    assert found["early_rate"] == pytest.approx(rates[:2].mean())
    assert found["late_rate"] == pytest.approx(rates[-2:].mean())
    assert found["late_minus_early"] == pytest.approx(found["late_rate"] - found["early_rate"])
    assert found["late_over_early"] == pytest.approx(found["late_rate"] / found["early_rate"])
    assert found["known_late_over_early"] == pytest.approx(73.5 / 67.4)
