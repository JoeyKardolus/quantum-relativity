"""A box's opacity is a fixed log scale of its energy over the start's mean, never a rank."""
from __future__ import annotations

import numpy as np
import warp as wp

from physics.view.opacity import FAINTEST, PACKED, THINNEST, opacity


@wp.kernel
def _read(ratios: wp.array(dtype=wp.float32), solid: wp.array(dtype=wp.float32)):
    index = wp.tid()
    solid[index] = opacity(ratios[index])


def _opacity(ratios) -> np.ndarray:
    ratios = wp.array(np.asarray(ratios, dtype=np.float32), device="cpu")
    solid = wp.zeros(ratios.shape[0], dtype=wp.float32, device="cpu")
    wp.launch(_read, dim=ratios.shape[0], inputs=[ratios, solid], device="cpu")
    return solid.numpy().astype(float)


def test_denser_is_never_fainter_and_rises_inside_the_scale():
    ratios = np.geomspace(0.01, 100.0, 400)
    alpha = _opacity(ratios)
    assert np.all(np.diff(alpha) >= 0)
    inside = (ratios > THINNEST * 1.01) & (ratios < PACKED / 1.01)
    assert np.all(np.diff(alpha[inside]) > 0)


def test_the_scale_is_fixed_ends_and_start_mean_near_invisible():
    alpha = _opacity([0.01, THINNEST, 1.0, PACKED, 1e6, 0.0, np.nan, np.inf])
    assert np.allclose(alpha[[0, 1, 5, 6]], FAINTEST, rtol=1e-5)
    assert np.allclose(alpha[[3, 4, 7]], 1.0, rtol=1e-5)
    assert alpha[2] < 0.05


def test_log_of_opacity_is_linear_in_log_energy():
    steps = np.diff(np.log(_opacity([1.0, 2.0, 4.0, 8.0])))
    assert np.allclose(steps, steps[0], rtol=1e-4)
