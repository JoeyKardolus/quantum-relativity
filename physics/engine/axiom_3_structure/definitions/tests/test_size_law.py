"""A cell's length is a calm volume's quanta over its own: more energy, smaller, either sign."""
from physics.engine import line_cells, step, three_cells
from physics.engine.axiom_3_structure.definitions.size_law import size_of


def test_length_is_a_constant_over_the_quanta_for_volumes_and_joints_alike():
    assert size_of(648_000, 6) == 1.0 and size_of(-648_000, 6) == 1.0
    assert size_of(1_296_000, 6) == 0.5 and size_of(-324_000, 6) == 2.0
    assert size_of(41_472_000, 24) == 1.0 and size_of(0, 6) == float("inf")


def test_the_three_cells_sizes_follow_their_quanta_as_energy_crosses():
    store = three_cells()
    sizes = [size_of(int(held), 6) for held in store.cell_quanta.numpy()[list(line_cells())]]
    assert sizes == [2 / 3, 1.0, 2.0]
    for _ in range(100):
        step(store)
    left, joint, right = (size_of(int(held), 6) for held in store.cell_quanta.numpy()[list(line_cells())])
    assert left > 2 / 3 and right < 2.0 and joint < 1.0  # A grew, B shrank, the squeezed joint too
