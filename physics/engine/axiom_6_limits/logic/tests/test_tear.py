"""What gathered at a free place tears off as a volume, a seventh flipped to minus for each joint."""
import numpy as np

from physics.engine import first_volume
from physics.engine.axiom_3_structure.definitions.store import put
from physics.engine.axiom_6_limits.definitions.least_whirl import PARTS, least_whirl
from physics.engine.axiom_6_limits.logic.tear import tear, torn


def test_a_place_tears_off_only_once_every_part_of_it_could_be_a_whirl():
    store = first_volume(layers=2)
    middle, enough = store.side ** 3 // 2, int(PARTS * least_whirl(6)) + 1
    put(store.cell_quanta, middle + 2, enough - 1)
    assert torn(store) == {}
    put(store.cell_quanta, middle + 2, enough)
    assert set(torn(store)) == {middle + 2, middle + 1}  # the volume, and its joint to the first


def test_each_joint_is_a_seventh_of_the_new_volumes_either_side_and_plus_lost_is_minus_gained():
    store = first_volume(layers=2)
    middle, side = store.side ** 3 // 2, store.side
    beside, corner, alone = middle + 2 * side, middle + 2 * side + 2, middle - 2 * side * side - 2
    for place, gathered in ((beside, 700_000), (corner, 140_007), (alone, 80_000)):
        put(store.cell_quanta, place, gathered)
    made = torn(store)
    assert made == {beside: 500_000, beside - side: -100_000,          # its joint to the first volume
                    corner: 120_006, beside + 1: -100_000 - 20_001,    # both give to the joint between
                    alone: 80_000}                                     # no volume beside it: no joint
    assert sum(made.values()) + 2 * sum(-held for held in made.values() if held < 0) == 920_007
    tear(store)
    assert store.count == 6 and torn(store) == {}
    boxes = store.box_quanta.numpy()[:6].astype(np.int64).sum(axis=1)
    assert (boxes == store.cell_quanta.numpy()[store.own_cell.numpy()[:6]]).all()
    assert store.cell_laid.numpy()[[beside, beside - side, corner, beside + 1]].tolist() == [2, 0, 1, 2]


def test_a_volume_that_would_form_on_the_room_s_last_layer_stops_with_a_plain_message():
    import pytest
    store = first_volume(layers=1)
    put(store.cell_quanta, store.side ** 3 // 2 + 2, 700_000)
    with pytest.raises(RuntimeError, match="no room for another layer"):
        tear(store)
