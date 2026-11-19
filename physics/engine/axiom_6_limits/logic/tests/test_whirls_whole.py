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


def test_every_volume_is_a_whole_whirl_at_two_layers_and_nearly_every_one_at_three(grown, grown_patch):
    assert all(_whole(grown[0])) and len(_whole(grown[0])) > 27  # 29 of 29 at 19.7 turns
    whole = _whole(grown_patch)
    print(f"whole whirls at three layers, step {grown_patch.step}: {sum(whole)} of {len(whole)} volumes")
    assert sum(whole) > 0.9 * len(whole)  # 113 of 118 at 52.8 turns (2 of 85 while the crossing flooded)


# The mark is every volume: axiom 2, cells are swirls. Measured on the card, outside the suite (6
# across): 116 of 118 at three layers, 779 of 920 at six (271 turns), 1,002 of 2,106 at eight and 181
# of 3,014 at nine (717 turns). The inner layers stay whole through eight; the outer volumes do not.
@pytest.mark.xfail(strict=True, reason="113 of 118 volumes are whole whirls at three layers (52.8 turns), and "
                   "fewer further out: a thin outer volume holds too little for its whirl to stand")
def test_every_volume_is_a_whole_whirl_at_three_layers(grown_patch):
    assert all(_whole(grown_patch))


def test_the_leak_never_takes_a_volume_below_a_whirl_s_least_through_two_layers(grown):
    store, seen = grown
    print(f"the least a volume held: {seen['least'] / least_whirl(store.across):.2f} least whirls")
    assert seen["least"] >= 1.5 * least_whirl(store.across)  # measured: 1.71 least whirls, at every step
    assert store.collapses == 0  # and no joint is past its limit
