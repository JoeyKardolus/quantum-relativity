"""A cell's slot: -1 while calm, its slot while own, still right after another cell gave back."""
import pytest

from physics.engine import calm_cells
from physics.engine.axiom_3_structure.definitions.store import slot_of
from physics.engine.axiom_3_structure.logic.give_back import give_back
from physics.engine.axiom_3_structure.logic.take_own import take_own


def test_slot_of_follows_a_cell_through_taking_and_giving_back():
    store = calm_cells(2000, own_cap=3)           # 4 tiles of 512 cells
    assert [slot_of(store, cell) for cell in (0, 700, 1999)] == [-1, -1, -1]
    for cell in (700, 3, 1999):
        take_own(store, cell)
    assert [slot_of(store, cell) for cell in (700, 3, 1999, 4)] == [0, 1, 2, -1]
    assert store.count == 3 and store.open_tiles == 3
    assert give_back(store, 700)                  # the last slot's cell moves into slot 0
    assert [slot_of(store, cell) for cell in (700, 3, 1999)] == [-1, 1, 0] and store.count == 2


def test_a_full_own_set_stops_with_a_plain_message():
    store = calm_cells(4, own_cap=1)
    take_own(store, 2)
    with pytest.raises(RuntimeError, match="the own set is full"):
        take_own(store, 3)
    rows = calm_cells(2000, own_cap=3, tile_cap=1)
    take_own(rows, 700)
    take_own(rows, 701)                           # the same tile: no new row
    with pytest.raises(RuntimeError, match="the tile rows are full"):
        take_own(rows, 3)
    assert rows.count == 2 and rows.open_tiles == 1


def test_a_joint_is_dealt_like_a_volume_in_minus_to_the_quantum():
    store, volume = calm_cells(1, own_cap=1), calm_cells(1, own_cap=1)
    store.cell_quanta.fill_(-648_001)
    volume.cell_quanta.fill_(648_001)
    take_own(store, 0)
    take_own(volume, 0)
    boxes = store.box_quanta.numpy()
    assert int(boxes.sum()) == -648_001 and boxes.max() <= 0  # 0: on a line of the whirl
    assert (boxes == -volume.box_quanta.numpy()).all()
