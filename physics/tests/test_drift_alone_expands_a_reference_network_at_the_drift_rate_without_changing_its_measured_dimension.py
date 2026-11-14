"""Drift alone expands a reference network at the drift rate without changing its measured dimension.

WHAT IT PINS. The physics group end to end: physics/network builds a 2D
spatial reference network, physics/rules steps it with only drift on, and
physics/measure reads the expansion (mean_length, expansion_rate) and the
dimension (ball_dimension) back from the network alone.

HOW THE EXPECTED ANSWER IS DERIVED WITHOUT THE CODE UNDER TEST. Drift h
multiplies every length by (1 + h) per step, so the scale factor grows as
(1 + h)^t: rate ln(1 + h) per step, exactly; a central difference of
(1 + h)^t reads the rate as ((1 + h) - 1 / (1 + h)) / 2. Scaling
every length by one factor scales every distance by it, so the ball-growth
exponent cannot change: it stays the torus's dimension, 2, within 10%.

ONE FUNCTION. One case; the group's single test.

WHAT IT WAS PROVED AGAINST. Dropping the energy twice as fast as the length
grows in step.py lets the energy-length rule pull lengths back, so the growth
per step is no longer ln(1.01).

GUARDS. First version of the physics package, 2026-11-14.
"""
from __future__ import annotations

import numpy as np

from physics.measure import ball_dimension, expansion_rate, mean_length
from physics.network import spatial_reference
from physics.rules import RuleSettings, step


def test_drift_alone_expands_a_reference_network_at_the_drift_rate_without_changing_its_measured_dimension() -> None:
    """Rate = ln(1 + drift) per step, no acceleration, dimension still 2."""
    network, known = spatial_reference(8000, 2, seed=4)
    settings = RuleSettings(drift=0.01)
    history = [mean_length(network)]
    for _ in range(30):
        network = step(network, settings)
        history.append(mean_length(network))
    growth = np.diff(np.log(history))
    np.testing.assert_allclose(growth, np.log(1.01), rtol=1e-9)
    rate = expansion_rate(history)["rate"][1:-1]
    np.testing.assert_allclose(rate, (1.01 - 1 / 1.01) / 2, rtol=1e-6)
    measured = ball_dimension(network, seed=4)["dimension"]
    print(f"drift-only 2D: measured dimension {measured:.3f} known {known['dimension']}")
    assert abs(measured - known["dimension"]) <= 0.1 * known["dimension"]
