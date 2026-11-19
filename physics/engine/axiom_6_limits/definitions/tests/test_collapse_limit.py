"""The limit is one pull for the whole universe, and the universe's own start lies under it."""
import numpy as np

from physics.engine.axiom_3_structure.logic.first_volume import FIRST_QUANTA
from physics.engine.axiom_6_limits.definitions.collapse_limit import DEEP, TOP_SPEED, past_limit


def test_a_pull_is_past_the_limit_where_twice_it_is_at_least_the_top_speed_squared():
    assert DEEP == TOP_SPEED ** 2 / 2 == 6050.0  # calm volumes' pull
    assert past_limit(DEEP) and not past_limit(0.999 * DEEP)
    pulls = np.array([1.2, 150.0, 6049.0, 6050.0, 1e9])  # two calm volumes; a squeezed centre; the edge
    assert past_limit(pulls).tolist() == [False, False, False, True, True]


def test_the_first_volume_wrapped_in_a_full_grid_is_not_past_the_limit():
    wrapped = 2 * 1.5164 * FIRST_QUANTA / 648_000  # the pull law summed over a full grid: twice 1.516
    assert 5020 < wrapped < DEEP  # 5,025: the start is no black hole, with a fifth to spare
    assert not past_limit(wrapped) and (2 * wrapped) ** 0.5 > 100  # the least top speed: 100.2
