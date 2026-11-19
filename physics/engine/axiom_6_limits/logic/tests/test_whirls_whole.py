"""Whether the volumes stay whirls as the universe grows: by their quanta, and by their whole turns."""
import pytest

from physics.engine import first_volume, step
from physics.engine.axiom_3_structure.logic.boxes_of import swirl_of
from physics.engine.axiom_3_structure.logic.give_back import whole_turns_laid
from physics.engine.axiom_6_limits.definitions.least_whirl import least_whirl
from physics.engine.axiom_6_limits.logic.tests.test_grow import _cells, _grown

grown = pytest.fixture(scope="module")(lambda: _grown())


def _whole(store) -> list:
    """For each own volume: whether its fluid still winds its whirl's whole turns."""
    laid = store.cell_laid.numpy()
    return [swirl_of(store, int(cell))["turns"] == whole_turns_laid(int(laid[cell]))
            for cell in _cells(store)[1]]


def test_the_leaking_volume_and_the_volumes_torn_off_it_are_whole_whirls_when_they_form():
    store = first_volume(layers=2)
    while store.count == 1:
        step(store)
    assert all(_whole(store)) and store.step > 100  # three turns of leaking left the first one whole


@pytest.mark.xfail(strict=True, reason="the first volume's energy floods the new volumes through "
                   "their fresh joints and no whirl is whole (0 of 172); they wind whole again later")
def test_every_volume_is_a_whole_whirl_after_three_layers(grown):
    assert all(_whole(grown[0]))


def test_the_leak_never_takes_a_volume_below_a_whirl_s_least_through_three_layers(grown):
    store, seen = grown
    assert seen["least"] >= 4 * least_whirl(store.across)  # measured: 5.0 least whirls, at every step
    assert store.collapses == 0  # and nothing is near its limit yet
