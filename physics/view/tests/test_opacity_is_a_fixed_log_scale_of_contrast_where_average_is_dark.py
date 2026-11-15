"""A box's light is a fixed log scale of its density contrast; average density is nearly dark."""
from __future__ import annotations

import numpy as np
import warp as wp

from physics.view.light_code import BRIGHTEST, DIMMEST, light_code
from physics.view.opacity import AVERAGE_GLOW, PACKED, opacity

SIDE = 256.0


@wp.kernel
def _read(contrasts: wp.array(dtype=wp.float32), side: float, light: wp.array(dtype=wp.float32)):
    index = wp.tid()
    light[index] = opacity(contrasts[index], wp.float32(side))


@wp.kernel
def _code(lights: wp.array(dtype=wp.float32), codes: wp.array(dtype=wp.float32)):
    index = wp.tid()
    codes[index] = light_code(lights[index])


def _opacity(contrasts, side: float = SIDE) -> np.ndarray:
    contrasts = wp.array(np.asarray(contrasts, dtype=np.float32), device="cpu")
    light = wp.zeros(contrasts.shape[0], dtype=wp.float32, device="cpu")
    wp.launch(_read, dim=contrasts.shape[0], inputs=[contrasts, side, light], device="cpu")
    return light.numpy().astype(float)


def test_denser_is_never_fainter_and_rises_up_to_packed():
    contrasts = np.geomspace(0.01, 100.0, 400)
    light = _opacity(contrasts)
    assert np.all(np.diff(light) >= 0)
    assert np.all(np.diff(light[contrasts < PACKED / 1.01]) > 0)


def test_a_line_of_sight_of_average_boxes_is_nearly_dark():
    for side in (64.0, 256.0, 1024.0):
        average = _opacity([1.0], side)[0]
        assert np.isclose(average * side, AVERAGE_GLOW, rtol=1e-4)
        assert average < 1e-3
    assert _opacity([0.5])[0] < _opacity([1.0])[0] / 5


def test_the_scale_has_fixed_ends():
    light = _opacity([PACKED, 1e6, np.inf, 0.0, -1.0, np.nan])
    assert np.allclose(light[:3], 1.0, rtol=1e-5)
    assert np.all(light[3:] == 0.0)


def test_log_of_light_is_linear_in_log_contrast():
    steps = np.diff(np.log(_opacity([1.0, 2.0, 4.0, 8.0])))
    assert np.allclose(steps, steps[0], rtol=1e-4)


def test_the_light_byte_keeps_faint_light_and_rises_with_it():
    lights = np.array([0.0, DIMMEST / 2, 1e-5, 1e-4, 1e-2, 1.0, BRIGHTEST, BRIGHTEST * 10],
                      dtype=np.float32)
    source = wp.array(lights, device="cpu")
    codes = wp.zeros(len(lights), dtype=wp.float32, device="cpu")
    wp.launch(_code, dim=len(lights), inputs=[source, codes], device="cpu")
    coded = np.round(codes.numpy())
    assert list(coded[:2]) == [0.0, 0.0] and coded[-2] == coded[-1] == 255.0
    assert np.all(np.diff(coded[1:-1]) > 0) and coded[3] >= 1
