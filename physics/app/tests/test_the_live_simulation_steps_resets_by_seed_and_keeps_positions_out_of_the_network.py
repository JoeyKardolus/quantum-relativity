"""The live Simulation steps, a reset is reproducible by seed, and positions never reach the Network."""
from dataclasses import fields

import numpy as np

from physics.app import Simulation
from physics.network import Network

JOINTS = 300
NETWORK_FIELDS = {"joint_count", "pairs", "energy", "length", "charge", "causal", "merged_into", "held"}


def test_the_simulation_steps_and_a_reset_is_reproducible_by_seed():
    first, second = Simulation(JOINTS, seed=4), Simulation(JOINTS, seed=4)
    start_energy = first.network.energy.copy()
    for _ in range(3):
        first.advance()
        second.advance()
    assert first.steps == 3
    assert not np.array_equal(first.network.energy, start_energy)
    np.testing.assert_array_equal(first.network.energy, second.network.energy)
    first.reset(4, JOINTS)
    np.testing.assert_array_equal(first.network.energy, start_energy)
    assert first.steps == 0
    assert not np.array_equal(Simulation(JOINTS, seed=5).network.energy[:50], start_energy[:50])


def test_the_start_is_uneven_with_no_lump_placed():
    energy = Simulation(JOINTS, seed=2, unevenness=0.5).network.energy
    spread = energy.std() / energy.mean()
    assert 0.03 < spread < 0.07         # unevenness 0.5 of a 10% noise: about 5%


def test_display_positions_never_end_up_in_the_network():
    simulation = Simulation(JOINTS, seed=3)
    positions = simulation.layout()
    simulation.advance()
    assert positions.shape == (simulation.network.joint_count, 3)
    assert {field.name for field in fields(Network)} == NETWORK_FIELDS
    for name in NETWORK_FIELDS:
        value = getattr(simulation.network, name)
        assert not (isinstance(value, np.ndarray) and value.ndim == 2 and value.shape[1] == 3)


def test_settings_are_clamped_to_their_range_and_unknown_names_refused():
    simulation = Simulation(JOINTS, seed=1)
    simulation.apply({"energy_flow": 7.0, "speed": 0.0, "rewiring": 3.4})
    assert simulation.settings.energy_flow == 1.0
    assert simulation.speed == simulation.ranges["speed"][0]
    assert simulation.settings.rewiring == 3
    try:
        simulation.apply({"gravity": 1.0})
    except ValueError:
        return
    raise AssertionError("an unknown setting name was accepted")
