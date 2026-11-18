"""pack_cells: packing is quanta over length cubed, and the more packed cell's clock is slower."""
from __future__ import annotations

import numpy as np
import pytest

from physics.engine.axiom_3_structure.definitions.kinds import NO_CELL
from physics.engine.axiom_5_packing.logic.pack_cells import pack_cells
from physics.engine.axiom_5_packing.logic.tests.conftest import (BOXES, ENERGY_VOLUME, JOINT, changed_names, fill_even,
                                                                 set_cells, set_quanta, snapshot)


def test_the_start_s_volume_and_joint_get_their_hand_worked_packing_and_rate(state, settings):
    fill_even(state, settings)
    before = snapshot(state)
    pack_cells(state, settings)
    assert changed_names(before, state) == {"packing", "rate"}
    kind, packing, rate = state.host("kind"), state.host("packing"), state.host("rate")
    # a volume: 648,000 quanta, 10 long -> 648; a joint: 21,600 quanta, 300 long -> 0.0008
    assert packing[kind == ENERGY_VOLUME] == pytest.approx(648.0)
    assert packing[kind == JOINT] == pytest.approx(21600 / 300.0 ** 3)
    assert rate[kind == ENERGY_VOLUME] == pytest.approx(1.0 / 1.1, rel=1e-6)
    assert rate[kind == JOINT] == pytest.approx(1.0 / (1.0 + 0.0008 / 6480.0), rel=1e-6)


def test_more_quanta_is_smaller_more_packed_and_slower_for_both_kinds(state, settings):
    kind = state.host("kind")
    boxes = np.where(kind == ENERGY_VOLUME, 3000, 100) * (1 + np.arange(state.count) % 7)
    set_quanta(state, settings, boxes.astype(np.int64) * BOXES)
    pack_cells(state, settings)
    quanta, length, packing, rate = (state.host(name) for name in ("cell_quanta", "length", "packing", "rate"))
    for one_kind in (ENERGY_VOLUME, JOINT):
        order = np.argsort(quanta[kind == one_kind], kind="stable")
        assert np.all(np.diff(length[kind == one_kind][order]) <= 0)
        assert np.all(np.diff(packing[kind == one_kind][order]) >= 0)
        assert np.all(np.diff(rate[kind == one_kind][order]) <= 0)
    assert np.all((rate > 0) & (rate <= 1))
    # packing goes as the quanta to the fourth power: twice the quanta, 16 times the packing
    volumes = np.flatnonzero(kind == ENERGY_VOLUME)
    assert packing[volumes[1]] / packing[volumes[0]] == pytest.approx((quanta[volumes[1]] / quanta[volumes[0]]) ** 4,
                                                                       rel=1e-5)


def test_an_empty_cell_and_an_id_that_is_no_cell_have_packing_0_and_rate_1(state, settings):
    fill_even(state, settings)
    quanta, kind = state.host("cell_quanta"), state.host("kind")
    quanta[0] = 0
    set_quanta(state, settings, quanta)
    kind[1] = NO_CELL
    set_cells(state, "kind", kind)
    pack_cells(state, settings)
    assert state.host("packing")[:2].tolist() == [0.0, 0.0]
    assert state.host("rate")[:2].tolist() == [1.0, 1.0]
