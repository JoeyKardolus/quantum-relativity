"""Joint collapse keeps all energy and all space exactly and lowers the cell count.

Forcing a chain of three neighbouring joints plus one lone joint to give way
must merge the volumes they separate (union-find: a chain is one volume), keep
the total energy and the total space (energy volumes plus joints: the freed
space swells the joints around the merged volume), add the charges, drop the
joints inside merged volumes and leave no joint from a volume to itself. A full
step with a push limit just below where the most pulled joints settle collapses
them on its own and maps every old volume.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

from physics.cells import EngineSettings, new_engine, joint_collapse, apply_rules


def test_joint_collapse_keeps_energy_and_space():
    grid = new_engine(500, seed=2, unevenness=0.1).grid
    grid = replace(grid, charge=np.linspace(-1.0, 1.0, grid.cell_count))
    giving_way = np.zeros(len(grid.joints), dtype=bool)
    first = grid.joints[0, 0]
    giving_way[np.flatnonzero((grid.joints == first).any(axis=1))[:3]] = True
    giving_way[-1] = True
    links = sp.coo_matrix((np.ones(giving_way.sum()), tuple(grid.joints[giving_way].T)),
                          shape=(grid.cell_count, grid.cell_count))
    expected = connected_components(links, directed=False)[0]
    collapsed = joint_collapse(grid, giving_way)
    assert collapsed.cell_count == expected < grid.cell_count
    assert np.isclose(collapsed.total_energy(), grid.total_energy(), rtol=1e-12)
    assert np.isclose(collapsed.total_space(), grid.total_space(), rtol=1e-12)
    assert np.isclose(collapsed.charge.sum(), grid.charge.sum(), atol=1e-12)
    assert collapsed.volume.sum() < grid.volume.sum()
    assert (collapsed.joint_volume > 0).all()
    assert (collapsed.joints[:, 0] != collapsed.joints[:, 1]).all()
    assert len(np.unique(collapsed.joints, axis=0)) == len(collapsed.joints)
    assert collapsed.merged_into[grid.joints[0, 0]] == collapsed.merged_into[grid.joints[0, 1]]

    stepped = apply_rules(grid, EngineSettings.all_on(push_limit=0.99))
    assert stepped.cell_count < grid.cell_count
    assert len(stepped.merged_into) == grid.cell_count
    assert stepped.merged_into.max() == stepped.cell_count - 1
