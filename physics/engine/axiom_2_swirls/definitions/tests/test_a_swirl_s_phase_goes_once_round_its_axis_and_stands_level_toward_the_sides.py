"""The swirl's phase and the rings it is read on, against hand-worked values."""
from __future__ import annotations

import math

import numpy as np
import pytest
import warp as wp

from physics.engine.axiom_2_swirls.tests.ring_box import RINGS, ring_box_host, ring_size_host
from physics.engine.axiom_2_swirls.definitions.swirl_phase import swirl_phase, swirl_phase_host
from physics.engine.axiom_3_structure.definitions.kinds import BLOCK

wp.config.quiet = True


@wp.kernel
def _phases(axis: wp.int32, sense: wp.int32, phase: wp.array3d(dtype=wp.float64)):
    x, y, z = wp.tid()
    phase[x, y, z] = swirl_phase(axis, sense, x, y, z)


def _wrapped(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _turns_round(axis: int, sense: int, layer: int, ring: int) -> float:
    """How many whole turns the phase makes going once round one ring in the sense +1."""
    size = ring_size_host(ring)
    boxes = [ring_box_host(axis, layer, ring, index) for index in range(size + 1)]
    phases = [swirl_phase_host(axis, sense, *box) for box in boxes]
    return sum(_wrapped(later - earlier) for earlier, later in zip(phases, phases[1:])) / (2.0 * math.pi)


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_the_three_rings_of_a_layer_cover_it_once_and_each_goes_round_box_by_box(axis):
    assert RINGS == 3 and [ring_size_host(ring) for ring in range(RINGS)] == [20, 12, 4]
    for layer in range(BLOCK):
        seen = set()
        for ring in range(RINGS):
            boxes = [ring_box_host(axis, layer, ring, index) for index in range(ring_size_host(ring))]
            for here, there in zip(boxes, boxes[1:] + boxes[:1]):
                assert sum(abs(first - second) for first, second in zip(here, there)) == 1
                assert here[axis] == layer
            seen.update(boxes)
        assert len(seen) == BLOCK * BLOCK


@pytest.mark.parametrize("axis", [0, 1, 2])
@pytest.mark.parametrize("sense", [1, -1])
def test_the_phase_goes_exactly_once_round_the_axis_falling_the_way_the_swirl_goes(axis, sense):
    for layer in range(BLOCK):
        for ring in range(RINGS):
            assert _turns_round(axis, sense, layer, ring) == pytest.approx(-sense, abs=1e-12)


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_a_box_on_the_outside_has_the_phase_of_the_box_behind_it_so_nothing_is_carried_toward_a_side(axis):
    for x in range(BLOCK):
        for y in range(BLOCK):
            for z in range(BLOCK):
                behind = tuple(min(max(value, 1), BLOCK - 2) for value in (x, y, z))
                behind = tuple(value if index != axis else (x, y, z)[axis] for index, value in enumerate(behind))
                assert swirl_phase_host(axis, 1, x, y, z) == swirl_phase_host(axis, 1, *behind)
                other_layer = [x, y, z]
                other_layer[axis] = (other_layer[axis] + 1) % BLOCK
                assert swirl_phase_host(axis, 1, x, y, z) == swirl_phase_host(axis, 1, *other_layer)


@pytest.mark.parametrize("axis", [0, 1, 2])
@pytest.mark.parametrize("sense", [1, -1])
def test_the_kernel_s_phase_is_the_host_s(axis, sense):
    phase = wp.zeros((BLOCK, BLOCK, BLOCK), dtype=wp.float64, device="cpu")
    wp.launch(_phases, dim=(BLOCK, BLOCK, BLOCK), inputs=[axis, sense], outputs=[phase], device="cpu")
    host = np.array([[[swirl_phase_host(axis, sense, x, y, z) for z in range(BLOCK)] for y in range(BLOCK)]
                     for x in range(BLOCK)])
    assert np.allclose(phase.numpy(), host, atol=1e-15)
    assert np.all(np.abs(phase.numpy()) <= math.pi)
