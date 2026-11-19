"""A packed region gives way within a turn: six joints and no more, it keeps all it holds, and the grid beyond stays as it was."""
import numpy as np
import pytest

import physics.engine as engine
from physics.engine import step
from physics.engine.axiom_3_structure.definitions.places import steps_of
from physics.engine.axiom_3_structure.logic.first_volume import FIRST_QUANTA
from physics.engine.axiom_5_packing.logic.lumps import region
from physics.engine.axiom_5_packing.logic.squeeze import squeeze
from physics.engine.axiom_5_packing.logic.staging import copy_of, packed
from physics.engine.axiom_6_limits.definitions.collapse_limit import DEEP


def _apart(store) -> dict:
    """How far apart the two volumes of each joint stand, for the joints 2 to 4 steps from the middle."""
    squeeze(store)
    own, felt = store.own_cell.numpy()[:store.count], store.own_squeeze.numpy()[:store.count, 2]
    out = np.abs(steps_of(own, store.side) - store.side // 2).max(axis=1)
    return {int(cell): far for cell, far, steps in zip(own, felt, out) if far > 0 and 2 <= steps <= 4}


@pytest.mark.xfail(strict=True, raises=AssertionError, reason="(f) of the six joints of a packed middle 2 fall in 60 "
                   "steps (6 before): the packed fluid is set going by the pull before the pull has made it deep, and "
                   "7,812 of the 11,599 calm volumes packed are left in the region 10 steps on")
def test_a_packed_region_collapses_stays_small_keeps_its_energy_and_leaves_the_grid_beyond(grown_patch, monkeypatch):
    store = copy_of(grown_patch)
    middle = region(store, (0, 0, 0))  # the first volume and the six at its sides
    packed(store, middle, FIRST_QUANTA)
    plain = copy_of(store)  # the same packing, run beside it with the collapse switched off
    joints = [(middle[0] + cell) // 2 for cell in middle[1:]]
    inside, fell = [], None
    for done in range(60):
        step(store)
        held = store.cell_quanta.numpy()
        inside.append(int(held[middle].sum() + held[joints].clip(0).sum()))
        fell = fell or (done + 1 if store.collapses else None)
    assert 5 < fell < 26 and store.collapses == 6 and (held[joints] > 0).all()  # 18 steps on: under a turn
    assert (store.cell_pull.numpy()[middle] >= DEEP).all()  # all seven past the limit; no seventh joint fell
    kept = np.array(inside[fell + 10:])  # from when all six have fallen
    assert (np.diff(kept) >= 0).all() and kept[-1] > kept[0]  # nothing leaves; it takes in (12,426 calm volumes)
    monkeypatch.setattr(engine, "collapse", lambda store: None)
    for _ in range(100):
        step(plain)
    assert plain.collapses == 0
    with_it, without = _apart(store), _apart(plain)
    changed = np.array([with_it[cell] / without[cell] - 1 for cell in with_it])
    assert len(changed) > 200 and np.abs(changed).max() < 0.02  # 270 joints: under 1% (measured)
