"""A real run on which nothing collapses and nothing tears (the engine's `held` run): no relation
changes, so after 24 steps the volume graph is still exactly the start's wrapped 26-neighbour grid."""
from physics.engine import Settings, new_universe, read_cells, read_ways, step
from physics.measure import volume_graph

HELD = {"push_strength": 40.0, "tear_length": 200.0}
STEPS = 24


def test_after_the_held_run_the_volume_graph_is_still_the_reference_grid(device, lattice_graph):
    settings = Settings(**HELD)
    universe = new_universe(settings, cells_per_side=16, seed=1, device=device)
    for _ in range(STEPS):
        step(universe, settings)
    matrix, volumes = volume_graph(read_ways(universe), read_cells(universe)["kind"])
    assert len(volumes) == 512
    assert (matrix != lattice_graph(8)).nnz == 0
