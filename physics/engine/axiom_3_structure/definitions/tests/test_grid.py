"""Which cells lie next to which follows from their place, round a grid with no edge."""
import warp as wp

from physics.engine.axiom_3_structure.definitions.grid import axis_of, cell_beside, joints_of, place_of


def _places(grid: tuple, cell: int) -> tuple:
    return tuple(place_of(wp.vec3i(*grid), cell, axis) for axis in range(3))


def _volumes(grid: tuple) -> list[int]:
    cells = range(grid[0] * grid[1] * grid[2])
    return [cell for cell in cells if all(place % 2 == 0 for place in _places(grid, cell))]


def test_each_volume_has_six_joints_and_each_joint_lies_between_two_volumes():
    for grid in ((4, 4, 4), (8, 8, 8), (4, 8, 12)):
        volumes, shared = _volumes(grid), {}
        for volume in volumes:
            joints = joints_of(grid, volume)
            assert len(set(joints)) == 6 and all(axis_of(wp.vec3i(*grid), one) >= 0 for one in joints)
            for joint in joints:
                shared.setdefault(joint, []).append(volume)
        assert len(shared) == 3 * len(volumes) == 3 * grid[0] * grid[1] * grid[2] // 8
        for joint, pair in shared.items():  # each joint: two volumes, the ones beside it on its axis
            axis = axis_of(wp.vec3i(*grid), joint)
            beside = {cell_beside(wp.vec3i(*grid), joint, axis, way) for way in (-1, 1)}
            assert len(pair) == 2 and set(pair) == beside and len(beside) == 2


def test_the_grid_wraps_round_in_all_three_directions():
    grid = wp.vec3i(4, 4, 4)
    assert cell_beside(grid, 0, 0, -1) == 48 and cell_beside(grid, 48, 0, 1) == 0
    assert cell_beside(grid, 0, 1, -1) == 12 and cell_beside(grid, 0, 2, -1) == 3
    assert cell_beside(grid, 63, 2, 1) == 60 and cell_beside(grid, 63, 1, 1) == 51


def test_the_pattern_is_the_same_seen_from_any_volume():
    grid, seen = (8, 8, 8), set()
    for volume in _volumes(grid):
        here = _places(grid, volume)
        steps = []
        for joint in joints_of(grid, volume):  # where each joint lies from here, round the grid
            there = _places(grid, joint)
            steps.append(tuple((there[axis] - here[axis] + 4) % 8 - 4 for axis in range(3)))
        seen.add(tuple(steps))
    assert seen == {((-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1))}


def test_a_joint_s_axis_is_the_one_direction_its_place_is_odd_in():
    grid = wp.vec3i(4, 4, 4)
    assert [axis_of(grid, cell) for cell in (16, 4, 1)] == [0, 1, 2]
    assert [axis_of(grid, cell) for cell in (0, 5, 20, 21)] == [-1, -1, -1, -1]  # a volume; no cell
    assert axis_of(wp.vec3i(1, 4, 1), 1) == 1 and axis_of(wp.vec3i(1, 4, 1), 3) == 1  # a line
