"""check_pattern and measure_grid on a real small universe: every break the owner named is seen."""
from __future__ import annotations

import numpy as np
import warp as wp

from physics import engine

wp.config.quiet = True

HOLDS = {"touching": 0, "bare": 0, "one_sided": 0, "dead_links": 0, "loose_ways": 0, "missing_ways": 0,
         "shared_mouths": 0, "flickered": 0, "lone_joints": 0, "web_pieces": 1, "fewest_joints": 26,
         "crowded_mouths": 0, "inner_joints": 0}
FACES = (4, 10, 12, 14, 16, 22)


def _universe():
    return engine.new_universe(engine.Settings(pull_strength=1e-13), cells_per_side=8, seed=1, device="cpu")


def _strip(state, keep) -> None:
    """Drop every relation `keep(owner kind, other kind, code)` refuses, on both sides, and relink."""
    kind = state.host("kind")
    for cell in range(state.count):
        state.set_neighbours(cell, [(one, code) for one, code in state.neighbours(cell)
                                    if keep(int(kind[cell]), int(kind[one]), code)])
    engine.relink(state, range(state.count))


def test_the_start_is_a_perfect_pattern():
    assert engine.check_pattern(_universe()) == HOLDS


def test_volumes_that_keep_only_their_six_face_joints_are_seen_to_have_six_however_often_each_is_listed():
    state = _universe()
    _strip(state, lambda owner, other, code: owner == other or code in FACES)
    pattern = engine.check_pattern(state)
    assert pattern["fewest_joints"] == 6 and pattern["bare"] == 0 and pattern["inner_joints"] == 256
    kind = state.host("kind")
    for volume in np.flatnonzero(kind == engine.ENERGY_VOLUME).tolist():
        entries = state.neighbours(volume)
        for joint, code in list(entries):
            for more in range(27):
                if more != 13 and more not in FACES and (joint, more) not in entries and len(entries) < 26:
                    entries.append((joint, more))
                    state.set_neighbours(joint, state.neighbours(joint) + [(volume, 26 - more)])
                    break
        state.set_neighbours(volume, entries)
    engine.relink(state, range(state.count))
    again = engine.check_pattern(state)
    assert again["fewest_joints"] == 6 and again["one_sided"] == 0


def test_a_volume_stripped_of_every_joint_is_bare():
    state = _universe()
    volume = int(np.flatnonzero(state.host("kind") == engine.ENERGY_VOLUME)[3])
    for joint, code in state.neighbours(volume):
        state.set_neighbours(joint, [entry for entry in state.neighbours(joint) if entry[0] != volume])
    state.set_neighbours(volume, [])
    engine.relink(state, range(state.count))
    pattern = engine.check_pattern(state)
    assert pattern["bare"] == 1 and pattern["fewest_joints"] == 0


def test_joints_that_forget_every_other_joint_are_no_web():
    state = _universe()
    _strip(state, lambda owner, other, code: owner != other)
    pattern = engine.check_pattern(state)
    assert pattern["lone_joints"] == 448 and pattern["web_pieces"] == 448 and pattern["fewest_joints"] == 26


def test_an_even_grid_is_not_warped_and_a_lump_alone_shortens_the_way_to_it_by_under_two_in_a_hundred():
    state = _universe()
    kind = state.host("kind")
    for cell in np.flatnonzero(kind != 0).tolist():
        state.set_block("quanta", cell, np.full((6, 6, 6), 3000 * int(kind[cell]) if kind[cell] > 0 else -100,
                                                dtype=np.int32))
    from physics.engine.axiom_1_energy.logic.count_cells import count_cells
    from physics.engine.axiom_3_structure.logic.size_cells import size_cells
    settings = engine.Settings()
    count_cells(state)
    size_cells(state, settings)
    even = engine.measure_grid(state)
    assert even["warp"] == 1.0 and np.isclose(even["size"], 64 * 10.0 + 448 * 300.0)
    lump = int(np.flatnonzero(kind == 1)[5])
    state.set_block("quanta", lump, np.full((6, 6, 6), 300000, dtype=np.int32))
    count_cells(state)
    size_cells(state, settings)
    warped = engine.measure_grid(state)
    # a lump a hundred times fuller is a hundredth as long, but the joints round it are the way: 305.05 in 310
    assert 0.98 < warped["warp"] < 0.99 and warped["lump_share"] == 1 / 64
