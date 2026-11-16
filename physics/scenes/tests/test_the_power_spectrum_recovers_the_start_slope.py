"""The clustering spectrum of a field with power k**slope has that slope, and an even field has none."""
import numpy as np
import pytest

from physics.engine import uneven_field
from physics.scenes import power_spectrum


@pytest.mark.parametrize("slope", [-3.0, -2.0])
def test_the_spectrum_of_a_weak_ripple_has_the_asked_slope_within_ten_percent(slope):
    density = 1.0 + uneven_field(32, seed=3, slope=slope, unevenness=0.01)
    wavenumbers, power = power_spectrum(density)
    fitted, _ = np.polyfit(np.log(wavenumbers), np.log(power), 1)
    assert abs(fitted - slope) <= 0.1 * abs(slope)


def test_an_even_field_has_no_power():
    _, power = power_spectrum(np.full((16, 16, 16), 2.5))
    assert np.all(power == 0.0)
