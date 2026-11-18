"""Proved on the engine's own shape: a cubic grid whose nodes also touch diagonally (26 neighbours),
open at 41 a side and wrapped at 8 a side (a universe of 16 cells a side). True dimension 3."""
import pytest

from physics.measure.dimension import adjacency, grid_graph, measure_dimension, passes

SHAPES = {"open, 41 a side": (41, False), "wrapped, 8 a side": (8, True)}


def _measured(shape):
    side, wrapped = SHAPES[shape]
    return measure_dimension(adjacency(*grid_graph(side, 3, wrapped=wrapped, diagonals=True)))


@pytest.mark.parametrize("shape", sorted(SHAPES))
def test_the_ball_dimension_gives_three_within_the_pass_mark(shape):
    assert passes(_measured(shape)["ball_dimension"], 3)


@pytest.mark.xfail(strict=True, reason="not kept: on 26 neighbours the range rule starts the walk too early "
                                       "(3.745 and 4.003; dimension/CLAUDE.md); the window is the owner's choice")
@pytest.mark.parametrize("shape", sorted(SHAPES))
def test_the_spectral_dimension_gives_three_within_the_pass_mark(shape):
    assert passes(_measured(shape)["spectral_dimension"], 3)
