"""Tear law: an overstretched energy volume tears.

Contract: `tear_law(length, low, middle, high, tear_length)` (Warp func; float64, three
int64, float64) returns a bool: `length > tear_length` and each of the three pieces it
would become holds at least one quantum. Overstretched means too big, and a volume is
big because it holds too little energy (size law, axiom 3). `low`, `middle`, `high` are
the quanta in its two lowest, two middle and two highest layers across its swirl axis:
the low volume, the new joint and the high volume of tear_map.py. A volume whose middle
is empty has no plus to flip into a joint, and a piece holding nothing would be no cell,
so it does not tear until its fluid has spread.
"""
from __future__ import annotations

import warp as wp

wp.set_module_options({"fuse_fp": False})


@wp.func
def tear_law(length: wp.float64, low: wp.int64, middle: wp.int64, high: wp.int64, tear_length: wp.float64) -> bool:
    """Whether a volume is too long and each of its three pieces would hold something."""
    return length > tear_length and low > wp.int64(0) and middle > wp.int64(0) and high > wp.int64(0)
