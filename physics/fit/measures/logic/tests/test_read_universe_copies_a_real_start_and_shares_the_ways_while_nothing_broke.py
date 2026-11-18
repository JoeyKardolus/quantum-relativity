"""read_universe copies a real start's cells and ways to the host, and shares the ways with the read before while no collapse or tear happened."""
import numpy as np

from physics.engine import ENERGY_VOLUME, JOINT, Settings, new_universe, read_books, step
from physics.fit.measures.logic.read_universe import read_universe
from physics.fit.tests.needs_gpu import needs_gpu


@needs_gpu
def test_a_real_start_8_a_side_has_64_volumes_448_joints_and_13_ways_a_volume():
    settings = Settings(push_strength=40.0, tear_length=200.0)
    universe = new_universe(settings, 8, 1)
    cells = read_universe(universe)
    books = read_books(universe)
    assert (cells["side"], cells["count"]) == (8, 512)
    assert int((cells["kind"] == ENERGY_VOLUME).sum()) == 64 and int((cells["kind"] == JOINT).sum()) == 448
    assert int(cells["held"][cells["kind"] == ENERGY_VOLUME].sum()) == books["plus"]
    assert int(cells["held"][cells["kind"] == JOINT].sum()) == books["minus"]
    assert cells["start_volume_quanta"] == books["plus"] / 64
    joint, first, second = cells["ways"]
    assert len(joint) == 64 * 13 and (cells["kind"][joint] == JOINT).all()
    assert (cells["kind"][first] == ENERGY_VOLUME).all() and (cells["kind"][second] == ENERGY_VOLUME).all()
    assert np.allclose(cells["length"] * cells["held"], settings.size_constant, rtol=1e-6), "the size law, every cell"
    assert (cells["fell_into"] == -1).all() and (cells["collapses"], cells["tears"], cells["version"]) == (0, 0, 0)
    step(universe, settings)
    after = read_universe(universe, cells)
    assert after["ways"] is cells["ways"], "nothing collapsed or tore: the ways are not read again"
    assert not np.array_equal(after["held"], cells["held"]), "the fluid moved between cells"
    assert 0.0 < after["squeeze_max"] and 0.0 < after["packing_max"]
