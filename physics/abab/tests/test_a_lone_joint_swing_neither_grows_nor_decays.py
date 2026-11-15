"""A lone joint between two equal energy volumes keeps its swing size: no growth, no decay."""
import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq

from physics.abab import Settings, joint_rest_size, joint_squeeze, pull_potential, step
from physics.abab.tests._hand_built import hand_built
from physics.abab.joint_push import joint_push


def test_a_lone_joint_swing_neither_grows_nor_decays():
    """2000 steps through step(): the swing size from its energy stays within 1% of the first."""
    settings = Settings.all_on()
    rest = float(joint_rest_size(0.05))
    state = hand_built([0, 1, 0], [1.0, 0.05, 1.0], [1.0, 0.7 * rest, 1.0], [(0, 1), (2, 1)])
    squeeze = float(joint_squeeze(state, pull_potential(state), settings.attraction)[1])

    def force(size):
        return float(joint_push(np.array([size]), np.array([rest]), settings.joint_push, settings.push_power)[0]
                     - squeeze * size)

    balance = brentq(force, 1.01e-3 * rest, 10 * rest)

    def swing_size(size, rate):
        level = 0.5 * rate * rate - quad(force, balance, size, limit=200)[0]
        def above(point):
            return -quad(force, balance, point, limit=200)[0] - level
        return brentq(above, balance, 100 * rest) - brentq(above, 1.01e-3 * rest, balance)

    first = swing_size(0.7 * rest, 0.0)
    sizes = []
    for _ in range(2000):
        state = step(state, settings)
        assert state.energy[0] == state.energy[2] == 1.0
        sizes.append(swing_size(state.size[1], state.rate[1]))
    assert np.all(np.abs(np.array(sizes) / first - 1) < 0.01)
