"""Sign law: plus only pulls, minus only pushes.

Contract: the law of axiom 2 as three askers, each a Warp func with a host twin. Every
rule that makes or feels the pull, or pushes, asks here and nowhere else.
  makes_pull(kind)   true only for an energy volume: only plus energy is a source of the pull
  feels_pull(kind)   true only for an energy volume: only plus energy is moved by the pull
  pushes(kind)       true only for a joint: only minus energy pushes back
An id that is no cell does none of the three (axiom 1: empty does nothing). There is no
other sign: no charge, no cell that does both.
"""
from __future__ import annotations

import warp as wp

from physics.engine.axiom_3_structure.definitions.kinds import ENERGY_VOLUME, JOINT

wp.set_module_options({"fuse_fp": False})

_ENERGY_VOLUME = wp.constant(wp.int32(ENERGY_VOLUME))
_JOINT = wp.constant(wp.int32(JOINT))


@wp.func
def makes_pull(kind: wp.int32) -> bool:
    """Whether a cell of this kind is a source of the pull."""
    return kind == _ENERGY_VOLUME


@wp.func
def feels_pull(kind: wp.int32) -> bool:
    """Whether a cell of this kind is moved by the pull."""
    return kind == _ENERGY_VOLUME


@wp.func
def pushes(kind: wp.int32) -> bool:
    """Whether a cell of this kind pushes back."""
    return kind == _JOINT
