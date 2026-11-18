"""The reference grid of 8 a side has 512 nodes with 26 neighbours each; the mark passes 2.7 to 3.3 on both measures."""
import numpy as np

from physics.scenes.dimension.definitions.known import KNOWN_DIMENSION
from physics.scenes.dimension.definitions.pass_mark import passes
from physics.scenes.dimension.definitions.reference_grid import reference_grid


def test_every_node_of_the_reference_touches_26_others():
    matrix = reference_grid(8)
    assert matrix.shape == (512, 512)
    assert (np.asarray(matrix.sum(axis=1)).ravel() == 26).all()
    assert (matrix != matrix.T).nnz == 0


def test_both_measures_must_be_within_ten_percent_of_three():
    assert KNOWN_DIMENSION == 3
    assert passes({"end": {"ball_dimension": 2.8, "spectral_dimension": 3.2}})
    assert not passes({"end": {"ball_dimension": 2.6, "spectral_dimension": 3.0}})
    assert not passes({"end": {"ball_dimension": 3.0, "spectral_dimension": 3.4}})
