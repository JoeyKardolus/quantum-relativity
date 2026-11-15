"""Total energy is exact over 300 steps with every rule on."""
from physics.abab import Settings, new_state, step


def test_total_energy_is_exact_over_300_steps():
    """Seed 3, 500 volumes, unevenness 0.5, defaults: the total holds to rounding (1e-12 relative)."""
    state, _ = new_state(500, 3, 0.5)
    start = state.total_energy()
    settings = Settings.all_on()
    for _ in range(300):
        state = step(state, settings)
        assert abs(state.total_energy() - start) <= 1e-12 * start
