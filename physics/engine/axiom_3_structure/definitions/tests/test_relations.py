"""A joint's rows face the boxes whose middles lie in its part of a volume's side."""
import warp as wp

from physics.engine import three_cells
from physics.engine.axiom_3_structure.definitions.relations import box_along, joints_of


def test_the_boxes_in_a_part_are_those_whose_middles_lie_in_it():
    assert [box_along(0.0, 1.0, row, 6) for row in range(6)] == [0, 1, 2, 3, 4, 5]
    assert [box_along(0.5, 0.5, row, 6) for row in range(6)] == [3, 3, 4, 4, 5, 5]
    assert [box_along(0.25, 0.25, row, 6) for row in (0, 5)] == [1, 2]  # middles at 0.25 and 0.42
    assert box_along(0.125, 0.125, 0, 6) == -1  # too thin a part to hold a box's middle


def test_a_volume_s_joints_are_those_that_name_it():
    ends = three_cells().joint_ends.numpy()
    assert ends.tolist() == [[-1, -1, -1], [0, 2, 1], [-1, -1, -1]]
    assert joints_of(ends, 0) == [(1, 0)] and joints_of(ends, 2) == [(1, 1)] and joints_of(ends, 1) == []
