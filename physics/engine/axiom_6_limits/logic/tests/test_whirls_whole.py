"""Whether the volumes stay whirls as the universe grows: by their quanta, and by their whole turns."""
import pytest

from physics.engine import first_volume, step
from physics.engine.axiom_2_swirls.logic.read_swirl import whole_turns
from physics.engine.axiom_6_limits.definitions.least_whirl import least_whirl
from physics.engine.axiom_6_limits.logic.tests.test_grow import _grown

grown = pytest.fixture(scope="module")(lambda: _grown())


def _whole(store) -> list:
    """For each own volume: whether its fluid winds its whirl's whole turn in all three rings."""
    held = store.cell_quanta.numpy()[store.own_cell.numpy()[:store.count]]
    phase = store.box_phase.numpy()
    return [all(whole_turns(phase[slot], ring) == -1 for ring in range(3))
            for slot in range(store.count) if held[slot] > 0]


def test_the_leaking_volume_and_the_volumes_torn_off_it_are_whole_whirls_when_they_form():
    store = first_volume(layers=2)
    while store.count == 1:
        step(store)
    print(f"the first volumes tear off at step {store.step}")
    assert all(_whole(store)) and store.step > 50  # two turns of leaking left the first one whole


# The mark is every volume: axiom 2, cells are swirls. Measured (6 across): 27 of 153 whole at
# three layers (7.2 turns), 52 of 195 at six (25.7 turns), 116 of 259 at eight (48 turns); of
# the joints nine in ten are whole throughout. Six layers take 31 s: measured once, outside the suite.
@pytest.mark.xfail(strict=True, reason="2 of 85 volumes are whole whirls at three layers (6.5 turns): "
                   "what floods in through a one-box cap spreads both ways round the ring it enters")
def test_every_volume_is_a_whole_whirl_at_three_layers(grown):
    whole = _whole(grown[0])
    print(f"whole whirls at three layers, step {grown[0].step}: {sum(whole)} of {len(whole)} volumes")
    assert all(whole)


def test_the_leak_never_takes_a_volume_below_a_whirl_s_least_through_three_layers(grown):
    store, seen = grown
    print(f"the least a volume held: {seen['least'] / least_whirl(store.across):.2f} least whirls")
    assert seen["least"] >= 3 * least_whirl(store.across)  # measured: 3.05 least whirls, at every step
    assert store.collapses == 0  # and no joint is past its limit yet
