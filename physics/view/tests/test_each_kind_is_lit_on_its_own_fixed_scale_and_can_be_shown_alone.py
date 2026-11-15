"""Energy volumes and joints each read against their own start mean, so blue shows as well as red; keys pick a kind."""
from __future__ import annotations

import time
from types import SimpleNamespace

import numpy as np
import pytest
import warp as wp

from physics.view.box_points import (BLUE, ENERGY_VOLUME, GLOW, JOINT, RED, SHOW_BOTH, SHOW_ENERGY_VOLUMES,
                                     SHOW_JOINTS, BoxPoints)
from physics.view.light_code import LOG_DIMMEST, LOG_SPAN
from physics.view.opacity import AVERAGE_GLOW
from physics.view.start_reference import start_reference
from physics.view.stepper import Stepper

VOLUME_MEAN = 5.0
JOINT_MEAN = 0.2
SIDE = 20


def _view(positions, density, kind, border):
    return SimpleNamespace(positions=wp.array(np.asarray(positions, np.float32), device="cpu"),
                           density=wp.array(np.asarray(density, np.float32), device="cpu"),
                           kind=wp.array(np.asarray(kind, np.int32), device="cpu"),
                           border=wp.array(np.asarray(border, np.uint8), device="cpu"))


def _checkerboard(side: int = SIDE):
    """An even cube of boxes, alternating energy volume and joint, joints holding far less energy."""
    centres = np.stack(np.meshgrid(*[np.arange(side) + 0.5] * 3, indexing="ij"), -1).reshape(-1, 3)
    kind = np.where(np.floor(centres).astype(int).sum(1) % 2 == 0, ENERGY_VOLUME, JOINT)
    density = np.where(kind == ENERGY_VOLUME, VOLUME_MEAN, JOINT_MEAN)
    return _view(centres, density, kind, np.zeros(len(centres)))


def _light(colours: np.ndarray) -> np.ndarray:
    codes = colours[:, 3].astype(float)
    return np.where(codes > 0, np.exp(LOG_DIMMEST + LOG_SPAN * (codes - 1.0) / 254.0), 0.0)


def _tint(colours: np.ndarray) -> list:
    return [tuple(colour[:3]) for colour in colours.tolist()]


def test_the_start_reference_holds_each_kinds_own_mean():
    view = _checkerboard()
    assert np.allclose(start_reference(view.density, view.kind), [VOLUME_MEAN, JOINT_MEAN, SIDE])
    alone = start_reference(np.array([3.0, 3.0]), np.array([JOINT, JOINT]))
    assert np.allclose(alone, [1.0, 3.0, 2.0 ** (1.0 / 3.0)])
    with pytest.raises(ValueError):
        start_reference(np.ones(3), np.zeros(2))


def test_an_even_checkerboard_is_nearly_dark_and_blue_sums_to_as_much_light_as_red():
    view = _checkerboard()
    gathered = BoxPoints().boxes(view, start_reference(view.density, view.kind))
    light = _light(gathered.colours)
    tints = np.array([tint == RED for tint in _tint(gathered.colours)])
    assert len(light) == SIDE ** 3
    assert abs(light.sum() / SIDE ** 2 / AVERAGE_GLOW - 1.0) < 0.05
    assert abs(light[~tints].sum() / light[tints].sum() - 1.0) < 0.05


@pytest.mark.parametrize("kind", [ENERGY_VOLUME, JOINT])
def test_each_kind_is_brighter_the_denser_it_is_on_its_own_scale(kind):
    contrasts = np.geomspace(0.5, 12.0, 12)
    mean = VOLUME_MEAN if kind == ENERGY_VOLUME else JOINT_MEAN
    count = len(contrasts)
    positions = np.column_stack([np.arange(count) + 0.5, np.full(count, 0.5), np.full(count, 0.5)])
    view = _view(positions, contrasts * mean, np.full(count, kind), np.zeros(count))
    gathered = BoxPoints().boxes(view, np.array([VOLUME_MEAN, JOINT_MEAN, 16.0]))
    order = np.argsort(gathered.positions[:, 0])
    light = _light(gathered.colours[order])
    assert len(light) == count and np.all(np.diff(light) > 0)
    assert set(_tint(gathered.colours)) == {RED if kind == ENERGY_VOLUME else BLUE}


def test_a_shown_kind_alone_draws_only_its_boxes_and_bounds_stay_whole():
    view = _checkerboard(4)
    reference = start_reference(view.density, view.kind)
    volumes, joints, both = (BoxPoints().boxes(view, reference, shown)
                             for shown in (SHOW_ENERGY_VOLUMES, SHOW_JOINTS, SHOW_BOTH))
    assert set(_tint(volumes.colours)) == {RED} and len(volumes.positions) == 32
    assert set(_tint(joints.colours)) == {BLUE} and len(joints.positions) == 32
    assert len(both.positions) == 64
    assert volumes.low.tolist() == joints.low.tolist() == [0.5] * 3
    assert volumes.high.tolist() == joints.high.tolist() == [3.5] * 3
    with pytest.raises(ValueError, match="shown"):
        BoxPoints().boxes(view, reference, 0)


class _EvenSource:
    """A 4^3 checkerboard of boxes that never changes."""

    steps = 0

    def reset(self) -> None:
        pass

    def step(self) -> None:
        self.steps += 1

    def view(self):
        return _checkerboard(4)

    def stats(self) -> dict:
        return {}


def test_choosing_a_kind_redraws_even_while_paused():
    stepper = Stepper(_EvenSource(), paused=True, look=GLOW)
    try:
        _wait(stepper, lambda snapshot: len(snapshot.positions) == 64)
        stepper.show(SHOW_JOINTS)
        joints = _wait(stepper, lambda snapshot: len(snapshot.positions) == 32)
        assert set(_tint(joints.colours)) == {BLUE} and joints.paused
        with pytest.raises(ValueError):
            stepper.show(4)
    finally:
        stepper.stop()


def _wait(stepper, wanted, seconds: float = 10.0):
    ends = time.perf_counter() + seconds
    while time.perf_counter() < ends:
        snapshot = stepper.latest()
        if snapshot is not None and wanted(snapshot):
            return snapshot
        time.sleep(0.01)
    raise AssertionError(f"no snapshot in {seconds} s; error: {stepper.error!r}")
