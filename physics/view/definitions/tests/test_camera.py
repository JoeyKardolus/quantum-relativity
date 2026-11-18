"""The eye looks at the cell's middle, and the wheel never takes it out of its range."""
import numpy as np

from physics.view.definitions.camera import FARTHEST, NEAREST, camera, zoomed


def test_the_eye_looks_at_the_middle_with_square_directions():
    eye, right, up, forward = camera(0.7, -0.4, 3.0)
    assert np.allclose(eye + 3.0 * forward, 0.0)
    assert np.allclose([right @ up, right @ forward, up @ forward], 0.0)
    assert np.allclose([right @ right, up @ up, forward @ forward], 1.0)


def test_the_wheel_keeps_the_eye_in_its_range():
    distance = 2.6
    for notches in (400.0, 1.0e9, float("inf"), -400.0, -1.0e9, float("-inf"), 3.0):
        distance = zoomed(distance, notches)
        assert NEAREST <= distance <= FARTHEST
    assert np.isclose(zoomed(2.6, 1.0), 2.34) and np.isclose(zoomed(2.6, -1.0), 2.6 / 0.9)
