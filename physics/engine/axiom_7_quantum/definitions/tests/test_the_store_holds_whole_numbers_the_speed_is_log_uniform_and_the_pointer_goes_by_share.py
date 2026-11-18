"""Axiom 7's definitions: the phase array, the store of splits, the speed draw, the pointer, joined phases."""
from __future__ import annotations

import math

import numpy as np
import pytest

from physics.engine.axiom_1_energy.definitions.draw_uniform import draw_uniform_rows
from physics.engine.axiom_7_quantum.definitions.joined_phase import joined_phase_host
from physics.engine.axiom_7_quantum.definitions.phase_arrays import ARRAYS, HOST
from physics.engine.axiom_7_quantum.definitions.pointer_law import pointer_law
from physics.engine.axiom_7_quantum.definitions.quantum_settings import QuantumSettings
from physics.engine.axiom_7_quantum.definitions.speed_draw import speed_draw
from physics.engine.axiom_7_quantum.definitions.split_store import amount_of, empty_store

MANY = 20000


def _many_splits(quanta: tuple, cells: tuple = (0, 2, 8)) -> tuple:
    """The halves' table of `MANY` splits that all hold the same quanta in the same cells."""
    sources = np.repeat(np.arange(MANY, dtype=np.int64), len(cells))
    return sources, np.tile(np.array(cells, dtype=np.int64), MANY), np.tile(np.array(quanta, dtype=np.int64), MANY)


def test_every_box_has_a_phase_and_the_store_starts_unset():
    assert ARRAYS == {"phase": ("float64", "box", False)} and HOST == {"quantum": None}


def test_settings_defaults_and_what_they_refuse():
    assert (QuantumSettings().slowest_speed, QuantumSettings().fastest_speed) == (1.0, 64.0)
    for wrong in ({"slowest_speed": 0.0}, {"fastest_speed": -1.0}, {"slowest_speed": 3.0, "fastest_speed": 2.0}):
        with pytest.raises(ValueError):
            QuantumSettings(**wrong)


def test_an_empty_store_holds_no_split_and_no_fraction_anywhere():
    store = empty_store()
    assert all(len(store[name]) == 0 for name in ("source", "born", "pointer", "directed", "reacted", "part_cell"))
    assert all(column.dtype in (np.int64, np.bool_) for column in store.values() if isinstance(column, np.ndarray))
    assert set(store["step"].values()) == {0} and set(store["total"].values()) == {0} and store["detected"] == set()
    assert len(amount_of(store)) == 0


def test_a_split_s_amount_is_the_sum_of_its_halves():
    store = empty_store()
    store["source"] = np.array([3, 9], dtype=np.int64)
    store["part_source"] = np.array([3, 3, 9, 9, 9], dtype=np.int64)
    store["part_quanta"] = np.array([5, 7, 1, 1, 40], dtype=np.int64)
    assert amount_of(store).tolist() == [12, 42]


def test_the_speed_lies_in_the_range_and_every_doubling_is_as_likely():
    speeds = speed_draw(draw_uniform_rows(1, 0, 40, np.arange(MANY)), 1.0, 64.0)
    assert speeds.min() >= 1.0 and speeds.max() <= 64.0
    octaves = np.histogram(np.log2(speeds), bins=6, range=(0.0, 6.0))[0] / MANY
    assert np.all(np.abs(octaves - 1.0 / 6.0) < 0.012), octaves
    assert speed_draw(np.array([0.0, 0.5]), 2.0, 8.0).tolist() == pytest.approx([2.0, 4.0])
    with pytest.raises(ValueError):
        speed_draw(np.array([0.5]), 0.0, 2.0)


def test_the_pointer_lands_on_a_cell_with_that_cell_s_share_as_its_chance():
    part_source, part_cell, part_quanta = _many_splits((210, 30, 60))
    cells = pointer_law(part_source, part_cell, part_quanta, np.arange(MANY), draw_uniform_rows(7, 3, 41, np.arange(MANY)))
    shares = [float(np.mean(cells == cell)) for cell in (0, 2, 8)]
    assert shares == pytest.approx([0.70, 0.10, 0.20], abs=0.012), shares


def test_a_half_of_nothing_is_never_chosen_and_a_split_of_nothing_is_refused():
    part_source, part_cell, part_quanta = _many_splits((0, 5, 0))
    cells = pointer_law(part_source, part_cell, part_quanta, np.arange(MANY), draw_uniform_rows(7, 3, 41, np.arange(MANY)))
    assert set(cells.tolist()) == {2}
    assert pointer_law(part_source, part_cell, part_quanta, np.array([5]), np.array([0.999999])).tolist() == [2]
    with pytest.raises(ValueError):
        pointer_law(part_source, part_cell, part_quanta * 0, np.array([5]), np.array([0.5]))


def test_two_fluids_join_by_their_quanta_and_equal_opposite_waves_cancel_to_the_old_phase():
    few_join_many = joined_phase_host(np.array([1000]), np.array([0.3]), np.array([10]), np.array([2.0]))
    assert 0.3 < float(few_join_many[0]) < 0.32
    cancelled = joined_phase_host(np.array([10]), np.array([math.pi]), np.array([10]), np.array([0.0]))
    assert float(cancelled[0]) == math.pi
    into_empty = joined_phase_host(np.array([0]), np.array([1.0]), np.array([4]), np.array([-2.0]))
    assert float(into_empty[0]) == pytest.approx(-2.0)
