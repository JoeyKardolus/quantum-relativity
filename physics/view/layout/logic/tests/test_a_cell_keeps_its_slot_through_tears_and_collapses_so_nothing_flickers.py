"""A cell is drawn in the same place of the pattern for as long as it lives. Plain arrays, no engine."""
import numpy as np
import pytest

from physics.view.layout.logic.cell_pieces import cell_pieces
from physics.view.layout.logic.cell_slots import cell_slots
from physics.view.layout.logic.torn_lines import torn_lines
from physics.view.tests.grid_cases import inside, slice_faces, torn_lattice

SIDE = 4
LATTICE = SIDE ** 3


def test_a_tears_cells_are_stripes_of_the_torn_volumes_slot_in_order():
    kind, length, packing, lines, line_axis = torn_lattice(SIDE)
    assert lines == {0: [0, 64, 65]} and line_axis == {0: 1}
    slots = cell_slots(SIDE, kind, length, lines, line_axis)
    mine = slots["piece_slot"] == 0
    assert slots["piece_cell"][mine].tolist() == [0, 64, 65] and slots["slot_axis"][0] == 1
    assert np.allclose(slots["piece_low"][mine], [0, 30 / 62, 32 / 62])
    assert np.allclose(slots["piece_high"][mine], [30 / 62, 32 / 62, 1])
    assert slots["widths"][1, 0, 0, 0] == 62.0 and slots["widths"][0, 0, 0, 0] == 30.0
    pieces = cell_pieces(SIDE, kind, length, packing, lines, line_axis)
    row = {int(cell): corners for cell, corners in zip(pieces.cell, pieces.corners)}
    assert np.allclose(row[0][2], row[64][0]) and np.allclose(row[64][2], row[65][0]), "the stripes share their edges"
    tall = [float(row[cell][2, 1] - row[cell][0, 1]) for cell in (0, 64, 65)]
    assert tall[1] < tall[0] / 5 and abs(tall[0] - tall[2]) < 1e-3 * tall[0]
    assert pieces.kind[pieces.cell == 64] == -1 and len(pieces.cell) == LATTICE + 2
    faces, _ = slice_faces(pieces, 0)
    points = np.random.default_rng(3).uniform(pieces.low[:2], pieces.high[:2], size=(3000, 2))
    assert inside(points, faces).max() == 1


def test_later_tears_go_right_after_the_volume_that_tore():
    kind = np.concatenate([np.ones(LATTICE), [-1, 1, -1, 1, -1, 1]]).astype(int)
    torn_from = np.full(LATTICE + 6, -1)
    torn_from[[64, 65]], torn_from[[66, 67]], torn_from[[68, 69]] = 0, 0, 65
    axis = np.full(LATTICE + 6, 2)
    lines, line_axis = torn_lines(LATTICE, kind, torn_from, kind.copy(), axis)
    assert lines == {0: [0, 66, 67, 64, 65, 68, 69]}, "V keeps the low piece; each pair lies after its own volume"
    assert line_axis == {0: 2}
    made = kind.copy()
    kind[[64, 67]] = 0
    assert torn_lines(LATTICE, kind, torn_from, made, axis)[0] == {0: [0, 66, 65, 68, 69]}, "gone cells leave the line"


def test_an_id_that_is_not_half_of_a_tear_is_refused_not_guessed():
    kind = np.concatenate([np.ones(LATTICE), [-1, 1]]).astype(int)
    torn_from = np.full(LATTICE + 2, -1)
    with pytest.raises(ValueError):
        torn_lines(LATTICE, kind, torn_from, kind.copy(), np.zeros(LATTICE + 2, dtype=int))
    torn_from[[64, 65]] = 0
    with pytest.raises(ValueError):
        torn_lines(LATTICE, kind, torn_from, -kind, np.zeros(LATTICE + 2, dtype=int))
    with pytest.raises(ValueError):
        torn_lines(LATTICE, kind[:-1], torn_from[:-1], kind[:-1], np.zeros(LATTICE + 1, dtype=int))


def test_a_cell_that_is_gone_leaves_its_slot_and_every_other_cell_stays_where_it_was():
    kind, length, packing, lines, line_axis = torn_lattice(SIDE)
    before = cell_slots(SIDE, kind, length, lines, line_axis)
    made = kind.copy()
    kind[[1, 2, 65]] = 0
    length[[1, 2, 65]] = 0.0
    torn_from = np.full(len(kind), -1)
    torn_from[LATTICE:] = 0
    lines, line_axis = torn_lines(LATTICE, kind, torn_from, made, np.ones(len(kind), dtype=int))
    after = cell_slots(SIDE, kind, length, lines, line_axis)
    where = dict(zip(before["piece_cell"].tolist(), before["piece_slot"].tolist()))
    assert set(after["piece_cell"].tolist()) == set(where) - {1, 2, 65}
    assert all(where[cell] == slot for cell, slot in zip(after["piece_cell"].tolist(), after["piece_slot"].tolist()))
    assert after["widths"][2, 0, 0, 1] == 0.0 and after["widths"][0, 0, 0, 2] == 0.0
    pieces = cell_pieces(SIDE, kind, length, packing, lines, line_axis)
    assert len(pieces.cell) == LATTICE - 1
    assert (np.diff(pieces.corners[:, [0, 1], 0], axis=1) >= 0).all(), "an emptied slot closes up, it does not fold"


def test_an_endless_cell_is_still_drawn_in_its_own_slot():
    kind, length, packing, lines, line_axis = torn_lattice(SIDE)
    length[2] = np.inf
    slots = cell_slots(SIDE, kind, length, lines, line_axis)
    assert np.isfinite(slots["widths"]).all() and slots["widths"][0, 0, 0, 2] == 30.0, "as long as the longest cell"
    assert 2 in slots["piece_cell"].tolist()
