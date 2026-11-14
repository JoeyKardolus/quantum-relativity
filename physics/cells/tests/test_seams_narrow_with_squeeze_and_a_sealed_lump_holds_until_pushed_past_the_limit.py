"""Seams are the tessellation's edges, narrow as their joints are squeezed, and seal a lump.

  edges     every edge of a cell lies on two of its faces, so 3 * seams equals half
            the faces' corners; every seam's three pairs are joints, before and
            after a step.
  narrow    with every joint at the same share f of its rest volume, f from 1 down
            to the push limit, no seam ever opens wider; 1 at rest, 0 from the seal
            point (depth 1 - f >= seal_point) on.
  closed    with every other joint at rest, a joint squeezed past the seal point
            (but not to the limit) closes its seams and only those: energy held by one of its volumes reaches the other by no path.
  sealed    a lump (one volume and its neighbours, energy tripled) with every joint
            touching it at depth 0.45 (seal 0.4, limit 0.5) keeps exactly its
            energy over 50 leaks while the rest of the grid moves; engine_stats
            counts it as a sealed lump; the rules collapse nothing there, until one
            of its joints is pushed to 0.3 of rest, past the limit: then it collapses.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from physics.cells import (EngineSettings, apply_rules, engine_stats, joint_rest, leak,
                           new_engine, rest_volume, seam_joints, seam_openness, step)

SETTINGS = EngineSettings.all_on()


def _at_share(grid, share):
    """The grid with every joint at `share` of its rest volume."""
    rest = joint_rest(grid.volume, grid.joints, SETTINGS.joint_rest)
    return replace(grid, joint_volume=share * rest)


def test_seams_are_the_edges_and_their_pairs_are_joints():
    engine = new_engine(300, seed=4, unevenness=0.2)
    assert 3 * len(engine.grid.seams) * 2 == engine.tessellation.face_size.sum()
    stepped = step(engine, SETTINGS)
    for grid in (engine.grid, stepped.grid):
        assert seam_joints(grid.joints, grid.seams, grid.cell_count).shape == (len(grid.seams), 3)
    assert np.array_equal(stepped.grid.seams, stepped.tessellation.seams)


def test_openness_falls_as_squeeze_rises():
    grid = new_engine(300, seed=5, unevenness=0.2).grid
    shares = np.linspace(1.0, SETTINGS.push_limit, 11)
    openness = np.array([seam_openness(_at_share(grid, share), SETTINGS) for share in shares])
    assert np.allclose(openness[0], 1.0)
    assert (np.diff(openness, axis=0) <= 1e-12).all()
    assert (np.diff(openness.mean(axis=1)) < 0).any()
    sealed = 1 - shares >= SETTINGS.seal_point - 1e-12
    assert np.allclose(openness[sealed], 0.0)
    assert (openness[~sealed] > 0).all()


def test_a_seam_whose_joints_are_past_the_seal_point_passes_nothing():
    grid = new_engine(300, seed=6, unevenness=0.2).grid
    rest = joint_rest(grid.volume, grid.joints, SETTINGS.joint_rest)
    squeezed = rest.copy()
    squeezed[0] = (1 - 0.45) * rest[0]
    grid = replace(grid, joint_volume=squeezed)
    openness = seam_openness(grid, SETTINGS)
    meets = (seam_joints(grid.joints, grid.seams, grid.cell_count) == 0).any(axis=1)
    assert meets.any() and (openness[meets] == 0).all() and (openness[~meets] > 0).all()
    source, target = grid.joints[0]
    alone = np.zeros(grid.cell_count)
    alone[source] = 1.0
    leaked = leak(grid.joints, grid.seams, alone, np.ones(grid.cell_count), openness, 0.5)
    assert leaked[target] == 0.0
    assert leaked[source] < 1.0 and np.isclose(leaked.sum(), 1.0, rtol=1e-15)


def test_a_sealed_lump_keeps_its_energy_until_pushed_past_the_limit():
    engine = new_engine(300, seed=7, unevenness=0.1)
    grid = engine.grid
    lump = np.unique(np.concatenate([[0], grid.joints[(grid.joints == 0).any(axis=1)].ravel()]))
    energy = grid.energy.copy()
    energy[lump] *= 3
    grid = replace(grid, energy=energy, volume=rest_volume(energy))
    rest = joint_rest(grid.volume, grid.joints, SETTINGS.joint_rest)
    touching = np.isin(grid.joints, lump).any(axis=1)
    squeezed = np.where(touching, (1 - 0.45) * rest, rest)
    grid = replace(grid, joint_volume=squeezed)
    openness = seam_openness(grid, SETTINGS)
    grid = replace(grid, seam_open=openness)
    pull = np.random.default_rng(7).uniform(0.1, 2.0, grid.cell_count)
    leaked = grid.energy
    for _ in range(50):
        leaked = leak(grid.joints, grid.seams, leaked, pull, openness, SETTINGS.leak)
    assert np.array_equal(leaked[lump], grid.energy[lump])
    assert not np.allclose(leaked, grid.energy)
    assert np.isclose(leaked.sum(), grid.energy.sum(), rtol=1e-13)
    stats = engine_stats(replace(engine, grid=grid))
    assert stats["lumps"] == 1 and stats["sealed_lumps"] == 1
    assert apply_rules(grid, SETTINGS).cell_count == grid.cell_count
    pushed = squeezed.copy()
    pushed[np.flatnonzero(touching)[0]] = 0.3 * rest[np.flatnonzero(touching)[0]]
    collapsed = apply_rules(replace(grid, joint_volume=pushed), SETTINGS)
    assert collapsed.cell_count < grid.cell_count
    assert len(np.unique(collapsed.merged_into[lump])) < len(lump)
