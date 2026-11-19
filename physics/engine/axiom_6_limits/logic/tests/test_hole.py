"""A packed region gives way within a turn, stays small and keeps all it holds; the grid beyond does not stay as it was."""
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


@pytest.fixture(scope="module")
def packed_middle(grown_patch):
    """The patch with the first volume and the six at its sides packed, 60 steps on; the same
    packing beside it, not yet run; what the region held and how many joints had fallen, step by step."""
    store = copy_of(grown_patch)
    middle = region(store, (0, 0, 0))
    packed(store, middle, FIRST_QUANTA)
    plain = copy_of(store)
    joints = [(middle[0] + cell) // 2 for cell in middle[1:]]
    inside, fallen = [], []
    for _ in range(60):
        step(store)
        held = store.cell_quanta.numpy()
        inside.append(int(held[middle].sum() + held[joints].clip(0).sum()))
        fallen.append(store.collapses)
    return store, plain, middle, joints, np.array(inside), np.array(fallen)


def test_a_packed_region_collapses_stays_small_and_keeps_its_energy(packed_middle):
    store, _plain, middle, joints, inside, fallen = packed_middle
    fell = int(np.argmax(fallen > 0)) + 1
    print(f"the first joint fell at step {fell}: {fallen[fell - 1]} joints then, {fallen[-1]} at the end")
    assert fell < 26 and (store.cell_quanta.numpy()[joints] > 0).all()  # 3 steps on: each one's pull is a step late
    assert fallen[fell + 10] == fallen[-1] < 60  # 54 joints fall at once and no more (6 before): the
    assert (store.cell_pull.numpy()[middle] >= DEEP).all()  # volumes a thin joint off are past the limit too
    kept = inside[fell + 10:]
    assert (np.diff(kept) >= 0).all() and kept[-1] > kept[0]  # nothing leaves; it takes in (10,018 to 10,042 calm volumes)


@pytest.mark.xfail(strict=True, reason="(7) the grid beyond a collapse is nearly as it was, not within the mark: of 249 joints 2 to 4 "
                   "steps out one distance differs by 2.2% from the same packing run with the collapse off (the "
                   "mark is 2%; up to 3.3 times while the uncollapsed middle poured out)")
def test_a_collapse_leaves_the_grid_beyond_as_it_was(packed_middle, monkeypatch):
    store, plain, _middle, _joints, _inside, _fallen = packed_middle
    plain = copy_of(plain)
    monkeypatch.setattr(engine, "collapse", lambda store: None)
    for _ in range(60):
        step(plain)
    assert plain.collapses == 0
    with_it, without = _apart(store), _apart(plain)
    changed = np.array([with_it[cell] / without[cell] - 1 for cell in with_it])
    assert len(changed) > 200 and np.abs(changed).max() < 0.02
