"""The standing swirl's phase and share, and the rings it is read on, against hand-worked values."""
from __future__ import annotations

import math

import numpy as np
import pytest
import warp as wp

from physics.engine.axiom_2_swirls.tests.ring_box import RINGS, ring_box_host, ring_size_host
from physics.engine.axiom_2_swirls.definitions.swirl_phase import swirl_mode_host, swirl_phase, swirl_phase_host
from physics.engine.axiom_2_swirls.definitions.swirl_share import swirl_share, swirl_share_host
from physics.engine.axiom_3_structure.definitions.kinds import BLOCK

wp.config.quiet = True


@wp.kernel
def _phases(axis: wp.int32, sense: wp.int32, turned: wp.int32, phase: wp.array3d(dtype=wp.float64),
            share: wp.array3d(dtype=wp.float64)):
    x, y, z = wp.tid()
    phase[x, y, z] = swirl_phase(axis, sense, turned, x, y, z)
    share[x, y, z] = swirl_share(axis, x, y, z)


def _wrapped(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _turns_round(axis: int, sense: int, turned: int, layer: int, ring: int) -> float:
    """How many whole turns the phase makes going once round one ring in the sense +1."""
    size = ring_size_host(ring)
    boxes = [ring_box_host(axis, layer, ring, index) for index in range(size + 1)]
    phases = [swirl_phase_host(axis, sense, turned, *box) for box in boxes]
    return sum(_wrapped(later - earlier) for earlier, later in zip(phases, phases[1:])) / (2.0 * math.pi)


def _box(axis: int, layer: int, first: int, second: int) -> tuple:
    """The box of one layer along `axis` with the two coordinates across it."""
    box = [first, second]
    box.insert(axis, layer)
    return tuple(box)


def _wave(axis: int, sense: int, turned: int, box: tuple) -> complex:
    return math.sqrt(swirl_share_host(axis, *box)) * complex(math.cos(swirl_phase_host(axis, sense, turned, *box)),
                                                             math.sin(swirl_phase_host(axis, sense, turned, *box)))


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
@pytest.mark.parametrize("turned", [0, 1])
def test_the_phase_goes_exactly_once_round_the_axis_falling_the_way_the_swirl_goes(axis, sense, turned):
    for layer in range(BLOCK):
        for ring in range(RINGS):
            assert _turns_round(axis, sense, turned, layer, ring) == pytest.approx(-sense, abs=1e-12)


def test_the_swirl_s_wave_is_the_slowest_standing_wave_of_a_row_of_six_so_it_only_turns_in_place():
    mode = [swirl_mode_host(place) for place in range(BLOCK)]
    assert mode == pytest.approx([-math.sin(math.radians(angle)) for angle in (75, 45, 15)]
                                 + [math.sin(math.radians(angle)) for angle in (15, 45, 75)])
    slow = 2.0 * (1.0 - math.cos(math.pi / BLOCK))
    for place in range(BLOCK):
        beside = [mode[other] for other in (place - 1, place + 1) if 0 <= other < BLOCK]
        assert sum(other - mode[place] for other in beside) == pytest.approx(-slow * mode[place], abs=1e-12)


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_a_whirl_holds_its_core_open_and_the_block_s_mean_share_is_one(axis):
    shares = [swirl_share_host(axis, x, y, z) for x in range(BLOCK) for y in range(BLOCK) for z in range(BLOCK)]
    assert sum(shares) / len(shares) == pytest.approx(1.0, abs=1e-12)
    assert min(shares) == pytest.approx(2.0 * math.sin(math.radians(15)) ** 2)
    assert max(shares) == pytest.approx(2.0 * math.sin(math.radians(75)) ** 2)
    assert swirl_share_host(axis, *_box(axis, 0, 2, 3)) == swirl_share_host(axis, *_box(axis, 5, 2, 3)) == min(shares)


@pytest.mark.parametrize("axis", [0, 1, 2])
@pytest.mark.parametrize("sense", [1, -1])
@pytest.mark.parametrize("turned", [0, 1])
def test_two_cells_that_mirror_each_other_hold_equal_waves_in_the_boxes_that_face_each_other(axis, sense, turned):
    for layer in (0, 3):
        for place in range(BLOCK):
            here = _wave(axis, sense, turned, _box(axis, layer, BLOCK - 1, place))
            there = _wave(axis, -sense, 1 - turned, _box(axis, layer, 0, place))        # the neighbour along the first
            assert abs(here - there) < 1e-12
            here = _wave(axis, sense, turned, _box(axis, layer, place, BLOCK - 1))
            there = _wave(axis, -sense, turned, _box(axis, layer, place, 0))            # the neighbour along the second
            assert abs(here - there) < 1e-12
    corner = _wave(axis, sense, turned, _box(axis, 2, BLOCK - 1, BLOCK - 1))
    assert abs(corner - _wave(axis, sense, 1 - turned, _box(axis, 2, 0, 0))) < 1e-12   # the neighbour across the edge


@pytest.mark.parametrize("axis", [0, 1, 2])
@pytest.mark.parametrize("sense", [1, -1])
def test_the_kernel_s_phase_and_share_are_the_host_s(axis, sense):
    for turned in (0, 1):
        phase = wp.zeros((BLOCK, BLOCK, BLOCK), dtype=wp.float64, device="cpu")
        share = wp.zeros((BLOCK, BLOCK, BLOCK), dtype=wp.float64, device="cpu")
        wp.launch(_phases, dim=(BLOCK, BLOCK, BLOCK), inputs=[axis, sense, turned], outputs=[phase, share], device="cpu")
        boxes = [(x, y, z) for x in range(BLOCK) for y in range(BLOCK) for z in range(BLOCK)]
        host = np.array([swirl_phase_host(axis, sense, turned, *box) for box in boxes]).reshape(BLOCK, BLOCK, BLOCK)
        assert np.allclose(np.exp(1j * phase.numpy()), np.exp(1j * host), atol=1e-12)
        assert np.all(np.abs(phase.numpy()) <= math.pi)
        assert np.allclose(share.numpy().reshape(-1), [swirl_share_host(axis, *box) for box in boxes], atol=1e-15)
