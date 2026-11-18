"""check_step on a hand-filled state: 0 when sizes are the law, and it sees a wrong length and a used dead block."""
from __future__ import annotations

import numpy as np

from physics.engine.axiom_3_structure.definitions.kinds import JOINT, NO_CELL
from physics.engine.axiom_3_structure.logic.size_cells import size_cells
from physics.engine.axiom_3_structure.tests.check_step import check_step
from physics.engine.axiom_3_structure.tests.conftest import default_settings, filled_state

ZERO = ("wrong_lengths", "dead_lengths", "dead_boxes", "dead_ids")


def test_sizes_by_the_law_and_no_dead_id_give_all_zero():
    state, settings = filled_state(), default_settings()
    found = check_step(state, settings)
    assert [found[key] for key in ZERO] == [0, 0, 0, 0]
    assert found["cells"] == 64 and 0.0 < found["warp"] <= 1.0 and found["size"] > 0.0


def test_a_length_not_set_by_the_energy_is_counted():
    state, settings = filled_state(), default_settings()
    length = state.length.numpy()
    length[3] *= np.float32(1.0001)
    state.length.assign(length)
    assert check_step(state, settings)["wrong_lengths"] == 1
    quanta = state.cell_quanta.numpy()
    quanta[5] += 1
    state.cell_quanta.assign(quanta)
    assert check_step(state, settings)["wrong_lengths"] == 2
    size_cells(state, settings)
    assert check_step(state, settings)["wrong_lengths"] == 0


def test_a_gone_cell_must_have_an_empty_block_and_no_length():
    state, settings = filled_state(), default_settings()
    joint = int(np.flatnonzero(state.host("kind") == JOINT)[0])
    boxes = int(np.count_nonzero(state.block("quanta", joint)))
    state.set_cell("kind", joint, NO_CELL)
    found = check_step(state, settings)
    assert (found["dead_ids"], found["dead_boxes"], found["dead_lengths"]) == (1, boxes, 1)
    state.set_block("quanta", joint, np.zeros((6, 6, 6)))
    state.set_cell("cell_quanta", joint, 0)
    size_cells(state, settings)
    found = check_step(state, settings)
    assert (found["dead_ids"], found["dead_boxes"], found["dead_lengths"], found["wrong_lengths"]) == (1, 0, 0, 0)
