"""(f) A black hole forms when a joint collapses: does one, by energy gathering alone, as a universe grows six layers?"""
import numpy as np
import pytest

from physics.engine import first_volume, step
from physics.engine.axiom_3_structure.definitions.places import steps_of
from physics.engine.axiom_6_limits.definitions.collapse_limit import DEEP, TOP_SPEED


@pytest.fixture(scope="module")
def six_layers():
    """A universe grown six layers out (420 steps, 4 s), with the deepest pull seen and the fullest volume at each new layer."""
    store, deepest, fullest, layers = first_volume(layers=7), 0.0, [], 0
    while layers < 6 and store.step < 600:
        step(store)
        deepest = max(deepest, float(store.cell_pull.numpy().max()))
        out = np.abs(steps_of(store.own_cell.numpy()[:store.count], store.side) - store.side // 2).max() // 2
        if out > layers:
            layers = out
            fullest.append(int(store.cell_quanta.numpy().max()) // 648_000)
    return store, deepest, fullest


def test_growth_by_layers_still_happens_and_nothing_is_past_the_limit(six_layers):
    store, deepest, fullest = six_layers
    assert len(fullest) == 6 and store.step < 500  # layers at 2.5, 10.9, 13.8, 15.0, 15.9 and 16.8 turns
    assert deepest < DEEP  # 3,966 at the first volume, as the second layer forms; the limit is 6,050


@pytest.mark.xfail(strict=True, reason="(f) no joint collapses by gathering: the deepest pull is 3,966 (the limit 6,050); "
                   "the fullest volume holds 1,654, 1,632, 749, 436, 255 and 205 calm volumes as the layers form: "
                   "energy spreads, it does not gather. A top speed under 89.1 (110 now) would put that pull past "
                   "the limit, but it is the first volume's own: it would be the hole, at 10.9 turns")
def test_a_joint_collapses_by_itself_where_energy_gathers(six_layers):
    store, deepest, _fullest = six_layers
    assert store.collapses > 0 or 2 * deepest >= TOP_SPEED ** 2
