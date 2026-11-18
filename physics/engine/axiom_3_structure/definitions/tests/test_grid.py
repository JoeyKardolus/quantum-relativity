"""Which cells lie next to which follows from their place: neighbours share exactly one joint."""
from physics.engine.axiom_3_structure.definitions.grid import joints_of
from physics.engine.axiom_3_structure.logic.lay_square import SQUARE, four_volumes


def test_each_volume_of_the_square_has_two_joints_and_neighbours_share_exactly_one():
    first, second, third, fourth = (set(joints_of(SQUARE, volume)) for volume in (0, 2, 6, 8))
    assert [len(joints) for joints in (first, second, third, fourth)] == [2, 2, 2, 2]
    assert first & second == {1} and first & third == {3} and second & fourth == {5}
    assert third & fourth == {7} and not first & fourth and not second & third  # not across


def test_a_line_s_joint_is_the_one_cell_between_its_two_volumes():
    assert joints_of((1, 3, 1), 0) == [1] and joints_of((1, 3, 1), 2) == [1]


def test_the_square_holds_four_volumes_four_joints_and_nothing_where_the_joints_meet():
    held = four_volumes().cell_quanta.numpy()
    assert (held[[0, 2, 6, 8]] > 0).all() and (held[[1, 3, 5, 7]] == -648_000).all() and held[4] == 0
