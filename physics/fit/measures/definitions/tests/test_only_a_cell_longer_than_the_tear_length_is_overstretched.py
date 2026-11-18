"""A cell is overstretched only when it is longer than the tear length; a cell holding nothing is endlessly long, so it is."""
import numpy as np

from physics.fit.measures.definitions.overstretched import overstretched


def test_longer_is_overstretched_and_exactly_at_the_length_is_not():
    length = np.array([0.0, 19.9, 20.0, 20.1, np.inf])
    assert overstretched(length, 20.0).tolist() == [False, False, False, True, True]
