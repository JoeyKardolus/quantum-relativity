"""Press law: how hard the pull presses two volumes together along one way.

Contract: `way_press_law(quanta_a, far_a, reach_a, quanta_b, far_b, reach_b, pull_joint)`
(Warp func, all float64) returns
  quanta_a quanta_b / (quanta_a + quanta_b)
      * ((far_a - pull_joint) / reach_a + (far_b - pull_joint) / reach_b)
Each bracket term is the pull's slope across one volume, toward the joint: how hard one
quantum of that volume is pulled at the joint. Their sum is how hard the two volumes are
pulled toward each other, and the factor in front is how much of that the joint between
them has to bear: two volumes pulled the same way at the same rate (falling together
down one even slope) press nothing, whatever each holds. Above 0 they are pressed
together, below 0 drawn apart. How the caller finds the numbers for volume a (b alike):
the far set is a's neighbours whose code, seen from a, equals `opposite(mouth_a)` (the
side of a away from the joint); `far_a` and `far_length` are the plain means of their
pull and length; an empty far set gives `far_a = pull[a]` and `far_length = 0`;
`reach_a = length_joint / 2 + length_a + far_length / 2`. A side whose reach is not
above 0 or is endless adds nothing, and so does a way with an empty volume.
"""
from __future__ import annotations

import warp as wp

wp.set_module_options({"fuse_fp": False})


@wp.func
def _slope(far: wp.float64, reach: wp.float64, pull_joint: wp.float64) -> wp.float64:
    """The pull's slope across one volume, toward the joint."""
    if not (reach > wp.float64(0.0)) or wp.isinf(reach):
        return wp.float64(0.0)
    return (far - pull_joint) / reach


@wp.func
def way_press_law(quanta_a: wp.float64, far_a: wp.float64, reach_a: wp.float64, quanta_b: wp.float64,
                  far_b: wp.float64, reach_b: wp.float64, pull_joint: wp.float64) -> wp.float64:
    """The press along one way: what the joint bears of its two volumes being pulled together."""
    both = quanta_a + quanta_b
    if not (both > wp.float64(0.0)):
        return wp.float64(0.0)
    return quanta_a * quanta_b / both * (_slope(far_a, reach_a, pull_joint) + _slope(far_b, reach_b, pull_joint))
