"""The start's unevenness of joints and of energy volumes are two separate settings.

Owner, 2026-11-16 (decision D): the starting unevenness of joints and of energy
volumes are two separate settings for the fit to search. new_universe on a small start
(3**3 cells of 8 boxes, seed 1, on the CPU): with joint_unevenness 0 (the default)
every joint box holds -1000; with joint_unevenness 0.3 the joint boxes differ (their
root mean square spread over their mean is between 0.2 and 0.4) while every energy
volume box keeps exactly the quanta it had, and the signs do not change.
"""
import numpy as np

from physics.engine import new_universe


def test_joint_unevenness_moves_only_the_joints():
    even = new_universe(3, 8, seed=1, device="cpu").quanta.numpy()
    uneven = new_universe(3, 8, seed=1, device="cpu", joint_unevenness=0.3).quanta.numpy()
    joints = -uneven[uneven < 0].astype(np.float64)
    spread = float(joints.std() / joints.mean())
    print(f"even joints {np.unique(even[even < 0]).tolist()}, uneven joints {joints.min():.0f}..{joints.max():.0f}, "
          f"relative spread {spread:.3f}")
    assert np.all(even[even < 0] == -1000)
    assert np.array_equal(np.sign(even), np.sign(uneven))
    assert np.array_equal(even[even > 0], uneven[uneven > 0])
    assert 0.2 < spread < 0.4
