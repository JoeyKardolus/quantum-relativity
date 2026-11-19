"""Two joined volumes never stand closer than their widths: the joint between always has some thickness."""
import numpy as np

from physics.engine import step
from physics.engine.axiom_3_structure.logic.shared_sides import joints_of
from physics.engine.axiom_5_packing.logic.lumps import fuller, grown, region
from physics.engine.axiom_5_packing.logic.squeeze import squeeze


def _gaps(store) -> tuple:
    """For every joint between two volumes: their distance, their two half widths, the joint's size."""
    squeeze(store)
    sizes = 648_000 / np.abs(store.cell_quanta.numpy()[store.own_cell.numpy()[:store.count]]).clip(1)
    joints, (first, second), _axis = joints_of(store)
    return store.own_squeeze.numpy()[joints, 2], (sizes[first] + sizes[second]) / 2, sizes[joints]


def test_in_a_grown_cluster_every_distance_is_the_half_widths_and_a_joint_thickness_above_0():
    store = grown(3, 6)
    apart, widths, thick = _gaps(store)
    assert len(apart) > 200 and (thick > 0).all() and np.allclose(apart, widths + thick)
    fresh = thick.min()
    fuller(store, region(store, (0, 0, 0)), 40)  # a hard squeeze at the middle, staged
    for _ in range(23):
        step(store)
    apart, widths, thick = _gaps(store)
    assert (apart > widths).all() and (thick > 0).all() and np.allclose(apart, widths + thick)
    assert 0 < thick.min() < fresh  # squeezed thinner (measured: 15.4 calm lengths), never to nothing
