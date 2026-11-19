"""A cell's number is its place: volumes at even steps, a joint at the odd step between its two."""
import warp as wp

from physics.engine.axiom_3_structure.definitions.places import (
    beside, joint_axis, joint_ends, steps_of)


@wp.kernel
def _read(cells: wp.array(dtype=wp.vec4i), found: wp.array(dtype=wp.vec4i)):
    cell, side, axis, steps = cells[wp.tid()][0], cells[wp.tid()][1], cells[wp.tid()][2], cells[wp.tid()][3]
    found[wp.tid()] = wp.vec4i(joint_axis(cell, side), 0, 0, 0)
    if side > 0:  # a store that keeps no places is asked for no neighbour
        ends = joint_ends(cell, side, axis)
        found[wp.tid()] = wp.vec4i(joint_axis(cell, side), ends[0], ends[1], beside(cell, side, axis, steps))


def _places(*cells: tuple) -> list:
    """For each (cell, side, axis, steps): its joint axis, its ends along `axis`, the cell `steps` on."""
    found = wp.zeros(len(cells), dtype=wp.vec4i, device="cpu")
    wp.launch(_read, dim=len(cells), device="cpu",
              inputs=[wp.array(cells, dtype=wp.vec4i, device="cpu"), found])
    return found.numpy().tolist()


def test_a_joint_s_axis_and_its_two_volumes_follow_from_its_place():
    side, volume = 5, 2 * 25 + 2 * 5 + 2  # the place 2, 2, 2
    assert steps_of(volume, side).tolist() == [2, 2, 2] and _places((volume, side, 0, 0))[0][0] == -1
    for axis, apart in enumerate((25, 5, 1)):
        assert _places((volume + apart, side, axis, 0))[0][:3] == [axis, volume, volume + 2 * apart]
    assert _places((7, 0, 0, 0))[0][0] == -1  # a store that keeps no places has no joints by place


def test_past_the_room_s_end_there_is_no_cell():
    on = [found[3] for found in _places((0, 5, 2, 2), (0, 5, 1, 2), (0, 5, 0, 4),
                                        (0, 5, 2, -2), (4, 5, 2, 2), (124, 5, 0, 2))]
    assert on == [2, 10, 100, -1, -1, -1]

