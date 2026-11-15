"""Every contact joins an energy volume and a joint, at the start and through splits and collapses."""
import numpy as np
import pytest

from physics.abab import KIND_A, KIND_B, Settings, new_state, step


def _abab(state) -> bool:
    """Every row is (energy volume, joint), and no row repeats."""
    kinds = state.kind[state.contacts]
    unique = len(np.unique(state.contacts, axis=0)) == len(state.contacts)
    return bool((kinds[:, 0] == KIND_A).all() and (kinds[:, 1] == KIND_B).all() and unique)


@pytest.mark.parametrize("overrides", [{}, {"push_limit": 0.85, "split_size": 1.5}])
def test_every_contact_joins_an_energy_volume_and_a_joint(overrides):
    """Defaults (splits) and a high push limit (collapses too): ABAB holds every step for 60 steps."""
    state, positions = new_state(343, 2, 0.5)
    assert positions.shape == (343, 3) and _abab(state)
    settings = Settings.all_on(**overrides)
    for _ in range(60):
        state = step(state, settings)
        assert _abab(state)
    assert state.splits > 0
