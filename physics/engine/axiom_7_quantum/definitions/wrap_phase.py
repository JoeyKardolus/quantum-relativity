"""Wrap phase: a phase is an angle, so a whole turn more or less is the same phase.

Contract: `wrap_phase(angle)` (Warp func, float64) returns the same angle in (-pi, pi];
`wrap_phase_host` is the host twin. Every phase is wrapped by it before it is stored,
and every phase step between two boxes before it is used.
"""
from __future__ import annotations

import math

import warp as wp

wp.set_module_options({"fuse_fp": False})

_PI = wp.constant(wp.float64(math.pi))
_TURN = wp.constant(wp.float64(2.0 * math.pi))


@wp.func
def wrap_phase(angle: wp.float64) -> wp.float64:
    """The angle brought into (-pi, pi]."""
    wrapped = angle - _TURN * wp.floor((angle + _PI) / _TURN)
    if wrapped <= -_PI:
        wrapped = wrapped + _TURN
    if wrapped > _PI:
        wrapped = wrapped - _TURN
    return wrapped


def wrap_phase_host(angle: float) -> float:
    """The angle brought into (-pi, pi] (host)."""
    wrapped = angle - 2.0 * math.pi * math.floor((angle + math.pi) / (2.0 * math.pi))
    if wrapped <= -math.pi:
        wrapped += 2.0 * math.pi
    if wrapped > math.pi:
        wrapped -= 2.0 * math.pi
    return wrapped
