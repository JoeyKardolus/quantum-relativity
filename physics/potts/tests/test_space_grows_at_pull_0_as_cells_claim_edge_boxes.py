"""Space grows: with the pull off and the joints pushing, the cells claim empty boxes at the edge.

Measured, not assumed: the test prints how much the occupied region grew in 60 steps
and asserts only that it grew, the bible's "space grows as cells claim empty boxes at
the edge". Also: no box was released, and the view source covers every occupied box.
"""
import numpy as np

from physics.potts import Settings, new_universe, stats, step, view_source
from physics.potts.tests._gpu import needs_gpu


@needs_gpu
def test_the_occupied_region_grows_by_claiming_edge_boxes():
    universe = new_universe(7, 4, seed=1, unevenness=0.3)
    settings = Settings.all_on(attraction=0.0)
    for _ in range(60):
        step(universe, settings)
    end = stats(universe, settings)
    print(f"occupied boxes {universe.start_occupied} -> {end['occupied']} (x{end['occupied_growth']:.3f}), "
          f"radius {end['radius']:.2f}, edge share {end['edge_share']:.4f}")
    assert end["occupied_growth"] > 1.0
    points = {name: array.numpy() for name, array in view_source(universe).items()}
    owner = universe.owner.numpy()
    assert points["positions"].shape == (end["occupied"], 3)
    places = (points["positions"] - 0.5).astype(int)
    assert np.array_equal(owner[places[:, 0], places[:, 1], places[:, 2]], points["cell"])
    assert np.array_equal(points["kind"], universe.kind.numpy()[points["cell"]])
    assert points["border"].any() and not points["border"].all()
