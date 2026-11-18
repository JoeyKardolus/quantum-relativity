"""Round a core a 4-volume lattice has 26 volumes one way out and 37 two; the held region and the depth radius stop where they should."""
import numpy as np
import pytest

from physics.scenes.black_hole.definitions.depth_radius import depth_radius
from physics.scenes.black_hole.definitions.held_shells import held_shells
from physics.scenes.black_hole.definitions.shell_ways import shell_ways


def test_a_volume_has_26_others_one_way_out():
    ways = shell_ways(4, 1)
    assert ways[1, 1, 1] == 0 and ways[3, 3, 3] == 2 and ways[2, 0, 1] == 1 and ways[0, 1, 1] == 1
    assert np.bincount(ways.ravel()).tolist() == [1, 26, 37]
    assert np.bincount(shell_ways(8, 4).ravel()).tolist() == [1, 26, 98, 218, 169]
    with pytest.raises(ValueError):
        shell_ways(4, 4)


def test_the_held_region_ends_at_the_first_boundary_that_lost_energy_and_never_at_the_last_read():
    assert held_shells([5.0, 1.0, 2.0, -1.0, 3.0]) == 2
    assert held_shells([-9.0, 1.0, 2.0, 3.0, 4.0]) == 3, "the core's skin is not asked; the last boundary is the limit"
    assert held_shells([5.0, 0.0, 2.0]) == 0 and held_shells([1.0]) == 0


def test_the_depth_radius_is_read_on_a_straight_line_between_two_shells():
    assert depth_radius([4.0, 2.0, 1.0, 0.0], [0.0, 10.0, 20.0, 30.0], 1.5) == (pytest.approx(15.0), 2)
    assert depth_radius([4.0, 2.0, 1.0, 0.0], [0.0, 10.0, 20.0, 30.0], 5.0) == (0.0, 0)
    radius, deep = depth_radius([4.0, 3.0, 3.0], [0.0, 10.0, 20.0], 1.0)
    assert deep == 2 and radius == 20.0, "never under the level: the last shell, flagged by the scene as the limit"
    with pytest.raises(ValueError):
        depth_radius([1.0], [0.0, 1.0], 0.5)
