"""(d) How the fall a fuller volume gives the others' fluid fades with distance: the patch frozen, the pull settled."""
import numpy as np
import pytest

from physics.engine.axiom_5_packing.logic.fall import fall
from physics.engine.axiom_5_packing.logic.lumps import fuller
from physics.engine.axiom_5_packing.logic.spread import spread
from physics.engine.axiom_5_packing.logic.staging import copy_of

CALM = 648_000


def _speed_gained(store, first: int, staged: bool) -> np.ndarray:
    """The speed each volume's fluid gains in a step, the pull handed on 300 times with nothing else moving."""
    store = copy_of(store)
    if staged:
        fuller(store, [first])
    for _ in range(300):
        spread(store)
    store.cell_flow.zero_()
    fall(store)
    return store.cell_flow.numpy() / np.maximum(store.cell_quanta.numpy(), 1)[:, None]


def _falls(store) -> np.ndarray:
    """(volumes out, distance in calm lengths, speed gained toward the middle) along the six rows from the middle."""
    side, first, held = store.side, store.side ** 3 // 2, store.cell_quanta.numpy()
    more, rows = _speed_gained(store, first, True) - _speed_gained(store, first, False), []
    for axis in range(3):
        for way in (-1, 1):
            far, stride = 0.0, way * side ** (2 - axis)
            for out in (1, 2, 3):
                cell, joint, before = first + 2 * out * stride, first + (2 * out - 1) * stride, first + 2 * (out - 1) * stride
                if held[cell] <= 0 or held[joint] >= 0:
                    break
                far += 0.5 * CALM / held[before] - CALM / held[joint] + 0.5 * CALM / held[cell]
                rows.append((out, far, -way * more[cell, axis]))
    return np.array(rows)


def test_the_middle_volume_made_fuller_pulls_every_volume_along_its_rows_toward_it(grown_patch):
    rows = _falls(grown_patch)
    assert len(rows) >= 10 and (rows[:, 2] > 0).all()  # 12 of 12, out to 78 calm lengths


@pytest.mark.xfail(strict=True, reason="(d) the fall fades as distance to the power -0.87, not -2, counted through the "
                   "cells' sizes (5 to 78 calm lengths), and as the power -3.0 counted in volumes out: the pull "
                   "is handed on a cell at a time whatever the cells' sizes, and one joint here is 50 lengths thick")
def test_the_pull_between_two_regions_fades_as_one_over_distance_squared(grown_patch):
    rows = _falls(grown_patch)
    power = np.polyfit(np.log(rows[:, 1]), np.log(rows[:, 2]), 1)[0]
    assert -2.3 < power < -1.7
