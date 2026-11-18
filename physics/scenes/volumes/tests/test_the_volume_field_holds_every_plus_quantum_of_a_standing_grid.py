"""While no volume has torn, the field on the start lattice holds every plus quantum the books count, before and after steps."""
from physics.engine import new_universe, read_books, step
from physics.scenes.tests._gpu import HELD, needs_gpu
from physics.scenes.volumes import count_lumps, observer_cells, volume_field


@needs_gpu
def test_the_field_sums_to_the_books_plus_quanta_at_the_start_and_after_three_steps():
    universe = new_universe(HELD, 8, seed=1)
    for taken in range(4):
        field, left_out = volume_field(universe)
        assert left_out == 0 and universe.tears == 0
        assert int(field.sum()) == read_books(universe)["plus"], f"step {taken}"
        fullest = int(observer_cells(universe, 1)[0])
        assert universe.host("cell_quanta")[fullest] == field.max()
        step(universe, HELD)
    assert count_lumps(universe, HELD)["lumps"] == int((field >= 2 * 648000).sum())
