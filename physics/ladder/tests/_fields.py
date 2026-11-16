"""Synthetic fields for the ladder's tests, and the device they run on.

DEVICE is Warp's "cpu" unless LADDER_TEST_DEVICE names another (the kernels are the
same source on CPU and CUDA; the default keeps the one GPU free for engine runs).
`cell_field` is a wrapped-cell pattern: energy volumes (plus) 7 boxes across in a
period of 10, wrapped in joints (minus), magnitudes about 2000 quanta with smooth
ripples of 3%, a gentle phase slope (no flow above the default tolerance), inside an
empty (zero) margin of one block.
"""
from __future__ import annotations

import os

import numpy as np
import warp as wp

wp.config.quiet = True
DEVICE = os.environ.get("LADDER_TEST_DEVICE", "cpu")


def cell_field(blocks: int = 6) -> tuple:
    """(quanta int32, phase float32) of a wrapped-cell pattern in a (8 blocks)^3 grid with a zero margin."""
    size = blocks * 8
    gx, gy, gz = np.meshgrid(np.arange(size), np.arange(size), np.arange(size), indexing="ij")
    volume = ((gx % 10) < 7) & ((gy % 10) < 7) & ((gz % 10) < 7)
    ripple = 1.0 + 0.03 * np.sin(2 * np.pi * gx / 37.0) * np.cos(2 * np.pi * gy / 29.0) * np.cos(2 * np.pi * gz / 31.0)
    magnitude = np.rint(2000.0 * ripple).astype(np.int32)
    quanta = np.where(volume, magnitude, -magnitude).astype(np.int32)
    inside = np.zeros_like(volume)
    inside[8:size - 8, 8:size - 8, 8:size - 8] = True
    quanta[~inside] = 0
    phase = (0.01 * gx + 0.005 * gy - 0.3).astype(np.float32)
    return quanta, phase


def wrapped_error(phase_a: np.ndarray, phase_b: np.ndarray) -> np.ndarray:
    """|phase_a - phase_b| wrapped into [0, pi]."""
    return np.abs(np.angle(np.exp(1j * (phase_a.astype(np.float64) - phase_b.astype(np.float64)))))
