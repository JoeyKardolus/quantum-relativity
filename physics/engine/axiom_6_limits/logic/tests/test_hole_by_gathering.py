"""A universe grown 40 turns: layers by the leak, whether the first volume holds (1), whether a joint collapses by gathering (7)."""
import numpy as np
import pytest

from physics.engine import first_volume, step
from physics.engine.axiom_3_structure.definitions.places import steps_of
from physics.engine.axiom_6_limits.definitions.collapse_limit import DEEP, TOP_SPEED


@pytest.fixture(scope="module")
def grown_40_turns():
    """A universe stepped 1,000 times (9 s): the turn each layer came at, the deepest pull seen, and
    what the first volume held at each whole turn, in calm volumes."""
    store, deepest, layers, first = first_volume(layers=7), 0.0, [], []
    for _ in range(1000):
        step(store)
        deepest = max(deepest, float(store.cell_pull.numpy().max()))
        out = np.abs(steps_of(store.own_cell.numpy()[:store.count], store.side) - store.side // 2).max() // 2
        if out > len(layers):
            layers.append(store.step / 25)
        if store.step % 25 == 0:
            first.append(int(store.cell_quanta.numpy()[store.side ** 3 // 2]) // 648_000)
    print(f"layers at {layers} turns; deepest pull {deepest:.0f}; collapses {store.collapses}")
    print(f"the first volume at 4, 8, 16, 24, 32, 40 turns: {[first[turn - 1] for turn in (4, 8, 16, 24, 32, 40)]}")
    return store, deepest, layers, first


def test_growth_by_layers_still_happens_and_nothing_is_past_the_limit(grown_40_turns):
    store, deepest, layers, _first = grown_40_turns
    assert len(layers) >= 2 and layers[0] < 3  # layers at 2.5 and 23.0 turns, by the leak alone (18.3, 22.6, 23.7 before were the first volume thrown outward)
    assert deepest < DEEP and store.collapses == 0  # the limit is 6,050


def test_the_first_volume_keeps_its_energy_for_sixteen_turns(grown_40_turns):
    _store, _deepest, _layers, first = grown_40_turns
    assert first[15] > 1640 and first[0] - first[15] < 10  # 1,653 at 4 turns, 1,649 at 16: it only leaks (1,632 at 10.9, 45 at 16.8 before)


def test_the_first_volume_holds_together_for_forty_turns(grown_40_turns):
    _store, _deepest, _layers, first = grown_40_turns
    assert first[39] > 1640 and first[15] - first[39] < 5  # 1,651 at 16 turns, 1,648 at 40: it only leaks (34 at 24 before)


@pytest.mark.xfail(strict=True, reason="(7) no joint collapses by gathering in 40 turns: the deepest pull seen is "
                   "1,867, under the limit of 6,050 (4,903 while the first volume poured into one neighbour)")
def test_a_joint_collapses_by_itself_where_energy_gathers(grown_40_turns):
    store, deepest, _layers, _first = grown_40_turns
    assert store.collapses > 0 or 2 * deepest >= TOP_SPEED ** 2
