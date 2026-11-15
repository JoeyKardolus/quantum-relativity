"""The pull solve fails loud: a measured volume that is not finite and positive stops it, never a silent potential."""
from dataclasses import replace

import numpy as np
import pytest

from physics.cells import new_engine, pull_potential


@pytest.mark.parametrize("bad", [-0.5, 0.0, np.nan, np.inf])
def test_a_volume_that_is_not_finite_and_positive_raises_instead_of_a_potential(bad):
    grid = new_engine(60, 2, 0.5).grid
    volume = grid.volume.copy()
    volume[7] = bad
    with pytest.raises(RuntimeError, match="not finite and positive"):
        pull_potential(replace(grid, volume=volume))
