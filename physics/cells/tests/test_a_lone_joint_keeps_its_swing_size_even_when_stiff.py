"""A lone joint keeps its swing: no growth and no decay from the numerics, even far past the old step's limit."""
import numpy as np
import pytest
from scipy.integrate import quad
from scipy.optimize import brentq

from physics.cells import joint_push, swing_joints

REST, SQUEEZE, POWER = 1.0, 0.05, 3.0


def _swing_size(push: float):
    """Thickness range between the turning points of a swing, from its energy (exact force integral)."""
    def force(thickness):
        return float(joint_push(np.array([thickness]), np.array([REST]), push, POWER)[0]
                     - SQUEEZE * thickness)

    balance = brentq(force, 1.01e-3 * REST, 10 * REST)

    def potential(thickness):
        return -quad(force, balance, thickness, limit=200)[0]

    def size(thickness, rate):
        level = 0.5 * rate * rate + potential(thickness)
        low = brentq(lambda point: potential(point) - level, 1.01e-3 * REST, balance)
        high = brentq(lambda point: potential(point) - level, balance, 100 * REST)
        return high - low

    return size


@pytest.mark.parametrize("push, start", [(0.1, 1.2), (0.5, 0.45), (2.0, 0.3)])
def test_a_lone_joint_keeps_its_swing_size_within_one_percent_for_a_thousand_steps(push, start):
    """Stiffness at the thin turning point 0.1 .. 250; the old explicit step went unstable past 4."""
    size = _swing_size(push)
    thickness, rate = np.array([start]), np.array([0.0])
    sizes = []
    for _ in range(1000):
        thickness, rate = swing_joints(thickness, rate, np.array([REST]), np.array([SQUEEZE]),
                                       push, POWER)
        sizes.append(size(thickness[0], rate[0]))
    first = size(start, 0.0)
    assert np.all(np.abs(np.array(sizes) / first - 1) < 0.01)
