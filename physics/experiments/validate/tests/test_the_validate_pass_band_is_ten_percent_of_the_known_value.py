"""The Validate pass band is ten percent of the known value.

WHAT IT PINS. physics/experiments/validate through `within`, the band every
Validate number is judged by: a measurement passes when it is within 10% of
the known value, relative, on either side.

HOW THE EXPECTED ANSWER IS DERIVED WITHOUT THE CODE UNDER TEST. The 10% band is
fixed in the bible before any run; the cases are worked by hand, including the
reported 1D miss (0.885 against 1) that must stay a miss.

ONE FUNCTION. Parametrized over the cases.

WHAT IT WAS PROVED AGAINST. Widening the band to 20%, which would turn the 1D
miss into a pass.

GUARDS. First version of the physics package, 2026-11-14.
"""
from __future__ import annotations

import pytest

from physics.experiments.validate._shared import within


@pytest.mark.parametrize(("measured", "known", "inside"), [
    (1.09, 1.0, True), (0.91, 1.0, True), (1.11, 1.0, False), (0.885, 1.0, False),
    (3.05, 3.0, True), (4.214, 4.0, True), (-3.2, -3.0, True), (-3.4, -3.0, False),
])
def test_the_validate_pass_band_is_ten_percent_of_the_known_value(measured, known, inside) -> None:
    """Inside the band exactly when within 10% of known."""
    assert within(measured, known) is inside
