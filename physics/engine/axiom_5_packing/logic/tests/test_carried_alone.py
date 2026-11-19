"""The same line: a lone fuller volume, and the crossing alone, where the pace is steady and the flow is booked like the quanta."""
import numpy as np
import pytest

from physics.engine.axiom_1_energy.logic.flow_inside import flow_inside
from physics.engine.axiom_4_crossing.logic.cross import cross
from physics.engine.axiom_5_packing.logic.lumps import FULLER
from physics.engine.axiom_5_packing.logic.line import VOLUMES
from physics.engine.axiom_5_packing.logic.tests.test_carried import CALM, _at, _held, _line, _turns


@pytest.mark.xfail(strict=True, reason="(2) a lone fuller volume off the line's middle does not stay put: it goes 1.48 "
                   "calm lengths toward the middle in 4 turns, pulled by the line's ten other volumes, more of them "
                   "on that side; a bench has ends, so only its middle is calm")
def test_a_lone_fuller_volume_stays_put():
    store, cells = _line((7,))
    start = _at(store, cells)
    _turns(store, 4)
    assert abs(_at(store, cells) - start) < 0.5


def test_with_the_crossing_alone_the_pace_is_steady_and_the_flow_is_kept_to_the_whole_number():
    store, cells = _line((1,), going=1.0)
    at = []
    for _ in range(6):
        _turns(store, 1, lambda one: (flow_inside(one), cross(one)))
        at.append(float((np.clip(_held(store, cells) - 1, 0, None) * np.arange(VOLUMES)).sum() / 3))
        assert store.cell_flow.numpy()[:, 1].sum() == FULLER * CALM  # friction holds energy back, the flow goes on
    assert np.all(np.diff(at) > 0.35) and np.all(np.diff(at) < 0.7)  # 0.60 in the first turn, then 0.53, 0.50, 0.47, 0.47, 0.49
