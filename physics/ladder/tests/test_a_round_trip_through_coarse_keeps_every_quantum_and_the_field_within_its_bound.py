"""FULL -> COARSE -> FULL on a quiet wrapped-cell field: totals and signs exact, field within the stated error.

Stated error (settings defaults): each box's quanta off by at most step - 1 of its
group, which the switch allows only when <= 1% of the group's smallest magnitude, so
<= 1% of the box's own; phase off by at most 0.02 rad. Promoting back to FULL keeps the
decoded values exactly. With the default quiet_steps (4), blocks leave FULL only on
the fourth quiet measurement in a row.
"""
from __future__ import annotations

import numpy as np

from physics.ladder import Level, LadderSettings, load_region, new_store, promote_blocks, read_region
from physics.ladder import switch_levels, total_quanta
from physics.ladder.tests._fields import DEVICE, cell_field, wrapped_error


def test_a_round_trip_through_coarse_keeps_every_quantum_and_the_field_within_its_bound():
    quanta, phase = cell_field()
    patient = new_store((6, 6, 6), device=DEVICE)
    load_region(patient, quanta, phase)
    assert [switch_levels(patient)["coarse"] for _ in range(4)] == [0, 0, 0, 4 ** 3]
    store = new_store((6, 6, 6), LadderSettings(quiet_steps=1), device=DEVICE)
    load_region(store, quanta, phase)
    plus, minus = int(quanta[quanta > 0].sum()), int(-quanta[quanta < 0].sum())
    assert total_quanta(store) == (plus, minus)
    report = switch_levels(store)
    assert report["coarse"] == 4 ** 3 and report["full"] == 0 and report["empty"] == 6 ** 3 - 4 ** 3
    assert total_quanta(store) == (plus, minus)
    coarse_q, coarse_p = read_region(store)
    space = quanta != 0
    assert np.array_equal(np.sign(coarse_q), np.sign(quanta))
    error = np.abs(coarse_q.astype(np.int64) - quanta)
    assert (error[space] <= 0.01 * np.abs(quanta[space])).all()
    assert wrapped_error(coarse_p, phase)[space].max() <= 0.02
    print(f"\nround trip: max quanta error {error.max()} of ~2000, mean {error[space].mean():.3f}; "
          f"max phase error {wrapped_error(coarse_p, phase)[space].max():.4f} rad")
    promote_blocks(store, np.argwhere(store.level != Level.EMPTY))
    assert (store.level[store.level != Level.EMPTY] == Level.FULL).all()
    full_q, full_p = read_region(store)
    assert np.array_equal(full_q, coarse_q) and np.array_equal(full_p, coarse_p)
    assert total_quanta(store) == (plus, minus)
