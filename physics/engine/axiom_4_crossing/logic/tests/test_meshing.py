"""A volume and its joint move alike where they face, like gears; two bare volumes meet half a turn apart."""
from physics.engine import line_cells, step, three_cells
from physics.engine.axiom_4_crossing.logic.meshing import meshing

A, JOINT, B = line_cells()


def _meshes(store) -> tuple:
    """The meshing of A with the joint, the joint with B, and of A with B as if nothing lay between."""
    return meshing(store, A, JOINT, 1), meshing(store, JOINT, B, 1), meshing(store, A, B, 1)


def test_a_volume_and_its_joint_agree_where_they_face_and_two_bare_volumes_do_not():
    store = three_cells(parts=(2, -2, 2))
    for steps in (0, 300):
        while store.step < steps:
            step(store)
        before, after, bare = _meshes(store)
        assert before[0] > 0.99 and after[0] > 0.99, (steps, before, after)  # measured: 1.0, 0.99999
        assert before[1] == after[1] == 1.0  # every row: both caps' fluid moves out from the jet
        assert bare[0] < -0.99 and bare[1] == 0.0, (steps, bare)  # half a turn apart, one out, one in


def test_gears_of_unequal_size_mesh_too():
    store = three_cells()  # A holds three times B's quanta: it turns three times as fast
    for _ in range(300):
        step(store)
    before, after, bare = _meshes(store)
    assert before[0] > 0.99 and after[0] > 0.98 and before[1] == after[1] == 1.0  # measured: 0.997, 0.992
    assert bare[0] < -0.9 and bare[1] == 0.0


def test_the_three_directions_mesh_alike():
    for along in range(3):
        store, (first, joint, second) = three_cells(parts=(2, -2, 2), along=along), line_cells(along)
        assert meshing(store, first, joint, along) == (1.0, 1.0)
        assert meshing(store, joint, second, along) == (1.0, 1.0)
