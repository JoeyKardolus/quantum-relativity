"""A tear on a real small universe: only that volume splits, across its own swirl axis, its middle flips to minus."""
from __future__ import annotations

import numpy as np
import pytest
import warp as wp

from physics import engine
from physics.engine.axiom_3_structure.definitions.direction_codes import step_of_host
from physics.engine.axiom_6_limits.definitions.held import Held
from physics.engine.axiom_6_limits.logic.tear import tear

wp.config.quiet = True


def _torn(axis: int):
    """A real 4-a-side universe in which volume 0, its swirl set about `axis`, has been torn."""
    settings = engine.Settings(pull_strength=1e-13)
    state = engine.new_universe(settings, cells_per_side=4, seed=1, device="cpu")
    state.set_cell("swirl_axis", 0, axis)
    state.set_cell("heat_quanta", 0, 3000)
    before = {"quanta": state.quanta.numpy().copy(), "phase": state.phase.numpy().copy(),
              "entries": {cell: state.neighbours(cell) for cell in range(state.count)},
              "sense": int(state.host("swirl_sense")[0]), "pull": float(state.host("pull")[0]),
              "kind": state.host("kind")}
    new_joint, new_volume = state.new_ids(2)
    held = Held()
    record = tear(state, held, 0, new_joint, new_volume)
    for cell in held.cells:
        state.set_neighbours(cell, held.entries[cell])
    for cell in held.quanta:
        state.set_block("quanta", cell, held.quanta[cell])
        state.set_block("phase", cell, held.phase[cell])
    engine.relink(state, held.cells)
    return state, record, before, new_joint, new_volume


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_it_tears_across_its_own_swirl_axis_and_every_quantum_is_kept_while_the_middle_flips(axis):
    state, record, before, joint, volume = _torn(axis)
    old = np.moveaxis(before["quanta"][0].astype(np.int64), axis, 0)
    assert record["axis"] == axis
    assert record["quanta"] == [int(old[0:2].sum()), int(old[2:4].sum()), int(old[4:6].sum())]
    assert int(state.block("quanta", 0).sum()) == record["quanta"][0] and (state.block("quanta", 0) > 0).all()
    assert int(-state.block("quanta", joint).sum()) == record["quanta"][1] and (state.block("quanta", joint) < 0).all()
    assert int(state.block("quanta", volume).sum()) == record["quanta"][2]
    assert state.torn == record["quanta"][1] and state.tears == 1
    assert state.host("kind")[[0, joint, volume]].tolist() == [1, -1, 1]
    assert int(state.host("fresh_quanta")[joint]) == record["quanta"][1]
    assert int(state.host("heat_quanta")[[0, joint, volume]].sum()) == 3000


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_the_pieces_are_swirls_about_the_same_axis_and_only_that_cell_s_fluid_changed(axis):
    state, record, before, joint, volume = _torn(axis)
    assert state.host("swirl_axis")[[0, joint, volume]].tolist() == [axis] * 3
    assert state.host("swirl_sense")[[0, joint, volume]].tolist() == [before["sense"]] * 3
    assert state.host("pull")[[joint, volume]].tolist() == [before["pull"]] * 2
    others = slice(1, joint)
    assert np.array_equal(state.quanta.numpy()[others], before["quanta"][others])
    assert np.array_equal(state.phase.numpy()[others], before["phase"][others])
    assert np.array_equal(state.host("kind")[1:joint], before["kind"][1:joint])


def test_the_new_joint_wraps_the_torn_sides_of_both_pieces_and_carries_nine_ways_between_them():
    state, record, before, joint, volume = _torn(0)
    for cell in (0, volume):
        assert {code for one, code in state.neighbours(cell)} == set(range(27)) - {13}
    ways = state.ways_of(joint)
    assert len(ways) == 9 and {(way_a, way_b) for slot, way_a, way_b, mouth_a, mouth_b in ways} == {(0, volume)}
    assert sorted(mouth_a for slot, way_a, way_b, mouth_a, mouth_b in ways) == [
        code for code in range(27) if step_of_host(code)[0] == 1]
    pattern = engine.check_pattern(state)
    assert pattern["web_pieces"] == 1 and pattern["inner_joints"] == 0
    assert not any(pattern[key] for key in ("touching", "bare", "one_sided", "dead_links", "loose_ways", "missing_ways",
                                            "shared_mouths", "flickered", "lone_joints"))
    # each piece keeps the 9 joints of its own side, the 8 round the middle and the new one; the 8 round the
    # middle now have two pieces on one side, so the mouth of the volume across each is shared by two ways
    assert pattern["fewest_joints"] == 18 and pattern["crowded_mouths"] == 8


def test_only_the_torn_volume_s_own_joints_gain_a_neighbour_and_no_relation_elsewhere_changes():
    state, record, before, joint, volume = _torn(0)
    touched = {one for one, code in before["entries"][0]} | {0}
    for cell, entries in before["entries"].items():
        if cell not in touched:
            assert state.neighbours(cell) == entries
        elif cell != 0:
            assert {one for one, code in state.neighbours(cell)} - {one for one, code in entries} <= {joint, volume}
