"""The census finds where energy has gathered away from the first volume, and reads the grown ball as it is."""
import numpy as np

from physics.engine.axiom_5_packing.logic.census import census, gatherings, line_of, whole_whirls
from physics.engine.axiom_5_packing.logic.line import calm_block


def test_two_fuller_volumes_in_a_calm_block_are_two_gatherings_fullest_first_and_an_even_block_has_none():
    store = calm_block((5, 5, 5), {(1, 1, 1): 4, (3, 3, 2): 3})
    seen = census(store)
    assert seen["fullest"] == [(4.0, (-1, -1, -1)), (3.0, (1, 1, 0))] and seen["gatherings"] == 2
    assert seen["over_calm"] == 2 and seen["volumes"] == 125 and seen["biggest"][:3] == [4.0, 3.0, 1.0]
    assert 2.0 < seen["apart"] < 6.0  # two volumes and a step off, through calm cells and thinner ones
    assert gatherings(np.ones((5, 5, 5))) == [] and whole_whirls(store) == 125


def test_a_fresh_volume_at_the_rim_is_no_gathering_and_the_first_volume_is_not_counted():
    block = np.zeros((5, 5, 5))
    block[2, 2, 2], block[2, 2, 3], block[0, 0, 0] = 1600.0, 0.5, 0.11  # the first, one it feeds, one just torn off
    assert gatherings(block) == []
    block[0, 0, 0] = 0.3
    assert gatherings(block) == [(0.3, (-2, -2, -2))]


def test_the_grown_ball_is_one_packed_volume_with_nothing_gathered_beside_it(grown_patch):
    seen = census(grown_patch)
    print(line_of(seen))
    assert seen["volumes"] == 111 and seen["layers"] == 3 and seen["share"] > 0.99 and seen["collapses"] == 0
    assert seen["over_calm"] == 0 and abs(seen["turns"] - 108.5) < 0.5
    assert 0 < whole_whirls(grown_patch) <= 111
