"""Push law: squeezed minus pushes back steeply, and only minus pushes.

Contract: two Warp funcs.
  push_law(joint_quanta, joint_length, push_strength)   (int64, float64, float64)
      a joint's push, `P (joint_quanta / joint_length) ** 2`, as float64. Since length
      is constant / quanta this is P quanta ** 4 / constant ** 2: steep in its energy.
      0 for an empty joint. The caller asks `pushes(kind)` (sign_law.py) first.
  squeeze_law(press, push)   (float64, float64)
      a joint's squeeze as float32, `max(press, 0) / push`: above 1 the press beats the
      push and the joint is squeezed. 0 for an empty joint (push 0).
"""
from __future__ import annotations

import warp as wp

wp.set_module_options({"fuse_fp": False})


@wp.func
def push_law(joint_quanta: wp.int64, joint_length: wp.float64, push_strength: wp.float64) -> wp.float64:
    """A joint's push: P times (quanta over length) squared."""
    if joint_quanta <= wp.int64(0) or not (joint_length > wp.float64(0.0)) or wp.isinf(joint_length):
        return wp.float64(0.0)
    thickness = wp.float64(joint_quanta) / joint_length
    return push_strength * thickness * thickness


@wp.func
def squeeze_law(press: wp.float64, push: wp.float64) -> wp.float32:
    """Press over push; 0 when nothing presses or the joint is empty."""
    if not (push > wp.float64(0.0)):
        return wp.float32(0.0)
    return wp.float32(wp.max(press, wp.float64(0.0)) / push)
