"""A cell is drawn in the same place of the pattern for as long as it lives; only a tear or a collapse changes the drawing's cells.

Staged with plain arrays (no engine), then a real engine run with hundreds of tears
and collapses (16 places, unevenness 0.3, ten times the default pull, as the engine's
own eventful run), checked after every step. Reported, not asserted, for that run: its
sizes come to differ by thousands of times, where the layout's solve may not settle.
"""
import numpy as np
import pytest

from physics.view import cell_pieces, cell_slots
from physics.view.tests.grid_cases import inside, slice_faces, uneven_lattice

SIDE = 4


def _torn():
    kind, length, packing = uneven_lattice(SIDE, spread=0.0)
    kind = np.concatenate([kind, [-1, 1]]).astype(np.int8)
    length = np.concatenate([length, [2.0, 30.0]])
    length[0] = 30.0
    return kind, length, np.ones(len(kind)), {SIDE ** 3: (0, 1), SIDE ** 3 + 1: (0, 1)}


def test_a_tears_cells_are_stripes_of_the_torn_volumes_slot_in_order():
    kind, length, packing, torn_from = _torn()
    slots = cell_slots(SIDE, kind, length, torn_from)
    mine = slots["piece_slot"] == 0
    assert slots["piece_cell"][mine].tolist() == [0, 64, 65] and slots["slot_axis"][0] == 1
    assert np.allclose(slots["piece_low"][mine], [0, 30 / 62, 32 / 62])
    assert np.allclose(slots["piece_high"][mine], [30 / 62, 32 / 62, 1])
    assert slots["widths"][1, 0, 0, 0] == 62.0 and slots["widths"][0, 0, 0, 0] == 30.0
    pieces = cell_pieces(SIDE, kind, length, packing, torn_from)
    row = {int(cell): corners for cell, corners in zip(pieces.cell, pieces.corners)}
    assert np.allclose(row[0][2], row[64][0]) and np.allclose(row[64][2], row[65][0]), "the stripes share their edges"
    tall = [float(row[cell][2, 1] - row[cell][0, 1]) for cell in (0, 64, 65)]
    assert tall[1] < tall[0] / 5 and abs(tall[0] - tall[2]) < 1e-3 * tall[0]
    assert pieces.kind[pieces.cell == 64] == -1 and len(pieces.cell) == SIDE ** 3 + 2
    faces, _ = slice_faces(pieces, 0)
    points = np.random.default_rng(3).uniform(pieces.low[:2], pieces.high[:2], size=(3000, 2))
    assert inside(points, faces).max() == 1


def test_a_cell_that_is_gone_leaves_its_slot_and_every_other_cell_stays_where_it_was():
    kind, length, packing, torn_from = _torn()
    before = cell_slots(SIDE, kind, length, torn_from)
    kind[[1, 2, 65]] = 0
    length[[1, 2, 65]] = 0.0
    after = cell_slots(SIDE, kind, length, torn_from)
    where = dict(zip(before["piece_cell"].tolist(), before["piece_slot"].tolist()))
    assert set(after["piece_cell"].tolist()) == set(where) - {1, 2, 65}
    assert all(where[cell] == slot for cell, slot in zip(after["piece_cell"].tolist(), after["piece_slot"].tolist()))
    assert after["widths"][2, 0, 0, 1] == 0.0 and after["widths"][0, 0, 0, 2] == 0.0
    pieces = cell_pieces(SIDE, kind, length, packing, torn_from)
    assert len(pieces.cell) == SIDE ** 3 - 1
    assert (np.diff(pieces.corners[:, [0, 1], 0], axis=1) >= 0).all(), "an emptied slot closes up, it does not fold"


def test_through_a_real_run_of_tears_and_collapses_no_cell_ever_changes_slot():
    wp = pytest.importorskip("warp")
    wp.config.quiet = True
    if wp.get_cuda_device_count() == 0:
        pytest.skip("the engine runs on a CUDA GPU")
    from physics.engine import Settings, new_universe, step
    state, settings = new_universe(16, unevenness=0.3), Settings(pull_strength=1e-6)
    where, drawn_twice, missing = {}, 0, 0
    for _ in range(12):
        step(state, settings)
        kind = state.kind_h[:state.count]
        slots = cell_slots(state.side, kind, state.host("length"), state.torn_from)
        cells = slots["piece_cell"].tolist()
        drawn_twice += len(cells) - len(set(cells))
        missing += int((kind != 0).sum()) - len(set(cells))
        for cell, slot in zip(cells, slots["piece_slot"].tolist()):
            assert where.setdefault(cell, slot) == slot, f"cell {cell} moved from slot {where[cell]} to {slot}"
        assert (slots["widths"] >= 0).all() and np.isfinite(slots["widths"]).all()
    assert state.tears > 100 and state.collapses > 100, "the run must be eventful"
    assert drawn_twice == 0 and missing == 0, "every living cell is drawn exactly once"
