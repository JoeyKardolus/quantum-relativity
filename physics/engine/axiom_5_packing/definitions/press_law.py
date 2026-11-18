"""Press law: how hard two volumes pull each other across the joint between them.

Contract: `way_press_law(quanta_a, pull_a, reach_a, quanta_b, pull_b, reach_b, pull_joint)`
(Warp func, all float64) returns
  quanta_a quanta_b / (quanta_a + quanta_b)
      * ((pull_joint - pull_a) / reach_a + (pull_joint - pull_b) / reach_b)
Each bracket term is the slope of the pull from the joint down into one of its two
volumes: how hard that volume pulls at the joint. Their sum is the pull across the
joint, and the factor in front is how much of it the joint has to bear. Two volumes on
one even slope (falling together toward a third mass) add nothing: one term rises by
what the other falls. Above 0 the two are pulled together and the joint is pressed;
below 0 (the joint lies deeper than its volumes) it is drawn apart.
`reach_a = (length_joint + length_a) / 2`, the middle of the joint to the middle of the
volume (b alike). A side whose reach is not above 0 or is endless adds nothing, and so
does a way with an empty volume.
Only the one pull field is read, as it stands at the joint and at its two volumes: no
quantum of a third cell is in it, and nothing is heard sooner than the pull carries it.
"""
from __future__ import annotations

import warp as wp

wp.set_module_options({"fuse_fp": False})


@wp.func
def _slope(pull_volume: wp.float64, reach: wp.float64, pull_joint: wp.float64) -> wp.float64:
    """The pull's slope from the joint down into one volume."""
    if not (reach > wp.float64(0.0)) or wp.isinf(reach):
        return wp.float64(0.0)
    return (pull_joint - pull_volume) / reach


@wp.func
def way_press_law(quanta_a: wp.float64, pull_a: wp.float64, reach_a: wp.float64, quanta_b: wp.float64,
                  pull_b: wp.float64, reach_b: wp.float64, pull_joint: wp.float64) -> wp.float64:
    """The press along one way: what the joint bears of its two volumes pulling each other."""
    both = quanta_a + quanta_b
    if not (both > wp.float64(0.0)) or not (quanta_a > wp.float64(0.0)) or not (quanta_b > wp.float64(0.0)):
        return wp.float64(0.0)
    return quanta_a * quanta_b / both * (_slope(pull_a, reach_a, pull_joint) + _slope(pull_b, reach_b, pull_joint))
