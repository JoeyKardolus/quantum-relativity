"""The drawing layout never lets two volumes that do not touch sit on top of each other."""
import numpy as np

from physics.app.settle_positions import settle_positions


def test_the_layout_keeps_volumes_that_do_not_touch_from_overlapping():
    # two energy volumes (0, 2) each touching joint 1, placed almost on top of each other
    positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.05, 0.02, 0.0]])
    contacts = np.array([[0, 1], [2, 1]])
    radii = np.array([0.5, 0.5, 0.5])
    settled = settle_positions(positions, contacts, radii, iterations=200)
    assert np.linalg.norm(settled[0] - settled[2]) > 0.9 * (radii[0] + radii[2])
