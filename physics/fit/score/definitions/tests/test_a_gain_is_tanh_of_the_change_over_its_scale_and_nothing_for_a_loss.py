"""A gain is tanh(change / scale): 0 for a loss or no change, never above 1 however large the change."""
import math

import pytest

from physics.fit.score.definitions.gain import gain


def test_known_gains():
    assert gain(0.2, 0.4) == pytest.approx(math.tanh(0.5)) == pytest.approx(0.462117, abs=1e-6)
    assert gain(0.0, 0.4) == 0.0 and gain(-3.0, 0.4) == 0.0
    assert gain(1e9, 0.1) == 1.0, "a runaway earns no more than the cap"
