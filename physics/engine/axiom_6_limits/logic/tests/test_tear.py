"""A tear makes two volumes and one joint with exact books; a joint along the side is cut in two."""
import numpy as np

from physics.engine import first_volume
from physics.engine.axiom_3_structure.logic.boxes_of import swirl_of
from physics.engine.axiom_3_structure.logic.give_back import whole_turns_laid
from physics.engine.axiom_6_limits.logic.tear import tear_volume


def _books(store) -> tuple:
    """The plus and the minus quanta, each cell's total checked against its own boxes."""
    held = store.cell_quanta.numpy()[:store.cells].astype(np.int64)
    boxes = store.box_quanta.numpy()[:store.count].astype(np.int64).sum(axis=1)
    assert (boxes == held[store.own_cell.numpy()[:store.count]]).all() and store.count == store.cells
    return int(held[held > 0].sum()), int(-held[held < 0].sum())


def _joints_lie_between_two_volumes(store) -> None:
    held, ends = store.cell_quanta.numpy()[:store.cells], store.joint_ends.numpy()[:store.cells]
    joints = np.flatnonzero(held < 0)
    assert len(joints) == store.tears + store.cuts and (held > 0).sum() == 1 + store.tears
    assert (ends[held > 0] == -1).all() and (ends[joints, 2] >= 0).all()  # only joints name cells
    assert (held[ends[joints, :2]] > 0).all() and (ends[joints, 0] != ends[joints, 1]).all()
    parts = store.joint_part.numpy()[joints]
    assert (parts[:, :, [1, 3]] > 0).all() and (parts[:, :, [0, 2]] + parts[:, :, [1, 3]] <= 1).all()


def test_a_tear_makes_two_volumes_and_one_joint_and_plus_lost_is_minus_gained():
    store = first_volume()
    plus, _none = _books(store)
    assert tear_volume(store, 0, 2) and store.cells == 3 and store.tears == 1
    halves, joint = _books(store)
    assert plus - halves == joint > 0 and halves + joint == plus == 323_999
    assert store.joint_ends.numpy()[:3].tolist() == [[-1, -1, -1], [-1, -1, -1], [0, 1, 2]]
    assert abs(joint / plus - 0.1424) < 0.001  # the two layers at the tear, of a whole whirl
    for half in (0, 1):  # each half a whole whirl again, the far one turned half round the line
        laid = whole_turns_laid(int(store.cell_laid.numpy()[half]))
        assert swirl_of(store, half)["turns"] == laid and swirl_of(store, half)["off_rule"] < 1
    assert store.cell_laid.numpy()[:3].tolist() == [0, 3, 0]


def test_a_joint_along_the_side_is_cut_in_two_keeping_its_minus():
    store = first_volume()
    tear_volume(store, 0, 2)
    minus = _books(store)[1]
    assert tear_volume(store, 0, 0) and store.cuts == 1 and store.cells == 6
    _joints_lie_between_two_volumes(store)
    ends, parts = store.joint_ends.numpy()[:6].tolist(), store.joint_part.numpy()[:6]
    assert ends[2] == [0, 1, 2] and ends[5] == [3, 1, 2] and ends[4] == [0, 3, 0]  # 3: the far half
    held = store.cell_quanta.numpy()
    assert held[2] + held[5] == -minus and abs(held[2] - held[5]) < 0.01 * minus
    assert parts[2].tolist() == [[0, 1, 0, 1], [0, 0.5, 0, 1]]  # half of the other volume's side
    assert parts[5].tolist() == [[0, 1, 0, 1], [0.5, 0.5, 0, 1]]
    assert tear_volume(store, 1, 0) and store.cuts == 1  # each part lies within one half: no cut
    ends, parts = store.joint_ends.numpy()[:8].tolist(), store.joint_part.numpy()[:8]
    assert ends[2] == [0, 1, 2] and ends[5] == [3, 6, 2] and (parts[[2, 5]] == [0, 1, 0, 1]).all()
    _joints_lie_between_two_volumes(store)
