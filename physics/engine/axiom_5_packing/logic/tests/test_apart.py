"""Two joined volumes never stand closer than their widths: the joint between always has some thickness."""
import numpy as np

from physics.engine import step
from physics.engine.axiom_3_structure.logic.joints_of import joints_of
from physics.engine.axiom_5_packing.logic.lumps import grown
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
    assert len(apart) > 150 and (thick > 0).all() and np.allclose(apart, widths + thick)  # 187 joints
    fresh = 60.0  # a joint as it tears off: a least whirl's quanta, 60 calm lengths thick
    for _ in range(12):  # the first volume, 1,044 calm volumes still, squeezes hard: no staging needed
        step(store)
    apart, widths, thick = _gaps(store)
    assert (apart > widths).all() and (thick > 0).all() and np.allclose(apart, widths + thick)
    assert 0 < thick.min() < fresh / 5  # squeezed thinner (measured: 5.3 calm lengths), never to nothing
