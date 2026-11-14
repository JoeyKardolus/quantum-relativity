"""No two energy volumes ever join unless a joint between them fell below its limit.

For 200 steps, with weak pull (attraction 0.002, where no joint reaches its
limit in 200 steps) and at the default settings (attraction 0.005: with no drift
balancing pull, space shrinks until joints give way and collapses happen),
each step's moved grid (move_space) is checked here independently: the joints
whose volume is below push_limit * joint_rest * (V_a + V_b) / 2 are found, and
step's merged_into must group old energy volumes exactly as the connected
components of those joints do. Two volumes with no chain of below-limit joints
between them must stay apart. The default run must see collapses and the
weak-pull run must see none, so both directions are exercised.
"""
from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

from physics.cells import CellSettings, build_grid, move_space, step

STEPS = 200


def _below_limit_groups(moved, settings):
    rest = settings.joint_rest * (moved.volume[moved.joints[:, 0]]
                                  + moved.volume[moved.joints[:, 1]]) / 2
    below = moved.joints[moved.joint_volume < settings.push_limit * rest]
    links = sp.coo_matrix((np.ones(len(below)), (below[:, 0], below[:, 1])),
                          shape=(moved.cell_count, moved.cell_count))
    return connected_components(links, directed=False)[1]


@pytest.mark.parametrize("attraction, collapses", [(0.002, False), (0.005, True)])
def test_joining_only_through_joints_below_their_limit(attraction, collapses):
    grid, _ = build_grid(600, seed=11, unevenness=0.1)
    settings = CellSettings.all_on(attraction=attraction)
    collapsed = 0
    for _ in range(STEPS):
        groups = _below_limit_groups(move_space(grid, settings), settings)
        stepped = step(grid, settings)
        pairs = set(zip(groups.tolist(), stepped.merged_into.tolist()))
        assert len(pairs) == len(set(groups.tolist())) == stepped.cell_count
        collapsed += grid.cell_count - stepped.cell_count
        grid = stepped
    assert (collapsed > 0) == collapses
