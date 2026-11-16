"""The slab (key C) keeps SLAB_BOXES box layers through the middle, moves held inside, and draws them bright by kind.

The fake source here stands in for the engine (physics.engine) in this test only;
the command always runs the real engine.
"""
from __future__ import annotations

import time
from types import SimpleNamespace

import numpy as np
import pytest
import warp as wp

from physics.view.box_points import (ENERGY_VOLUME, GLOW, JOINT, NEAREST, SHOW_ENERGY_VOLUMES, SLAB,
                                     SLAB_BLUE, SLAB_BORDER_LIFT, SLAB_BOXES, SLAB_DIMMEST, SLAB_RED,
                                     BoxPoints)
from physics.view.opacity import PACKED
from physics.view.stepper import Stepper
from physics.view.title_line import title_line

_REFERENCE = np.array([1.0, 1.0, 8.0])


def _checkerboard(side: int = 8, density=None, border=None):
    """An even cube of boxes, alternating energy volume and joint, at density 1 unless given."""
    centres = np.stack(np.meshgrid(*[np.arange(side) + 0.5] * 3, indexing="ij"), -1).reshape(-1, 3)
    kind = np.where(np.floor(centres).astype(int).sum(1) % 2 == 0, ENERGY_VOLUME, JOINT)
    density = np.ones(len(centres)) if density is None else density
    border = np.zeros(len(centres)) if border is None else border
    return SimpleNamespace(positions=wp.array(centres.astype(np.float32), device="cpu"),
                           density=wp.array(np.asarray(density, np.float32), device="cpu"),
                           kind=wp.array(kind.astype(np.int32), device="cpu"),
                           border=wp.array(np.asarray(border, np.uint8), device="cpu"))


@pytest.mark.parametrize("offset, layers, used", [
    (0, [2.5, 3.5, 4.5, 5.5, 6.5], 0),
    (-1, [1.5, 2.5, 3.5, 4.5, 5.5], -1),
    (3, [5.5, 6.5, 7.5], 3),
    (9, [5.5, 6.5, 7.5], 3),
    (-9, [0.5, 1.5, 2.5], -4)])
def test_the_slab_keeps_its_layers_around_the_middle_moved_and_held_inside(offset, layers, used):
    gathered = BoxPoints().boxes(_checkerboard(), _REFERENCE, look=SLAB, slice_offset=offset,
                                 slice_boxes=SLAB_BOXES)
    assert sorted(set(gathered.positions[:, 2].tolist())) == layers
    assert len(gathered.positions) == 64 * len(layers) and gathered.slice_offset == used
    assert gathered.slice_at == 4.5 + used
    assert gathered.low.tolist() == [0.5] * 3 and gathered.high.tolist() == [7.5] * 3


def test_the_slab_draws_kinds_bright_borders_lighter_and_packed_boxes_brightest():
    count = 8 ** 3
    density = np.ones(count)
    border = np.zeros(count)
    density[0], density[1] = PACKED, PACKED  # (0.5, 0.5, 0.5) energy volume, (0.5, 0.5, 1.5) joint
    border[2] = 1                            # (0.5, 0.5, 2.5) energy volume on a cell border
    gathered = BoxPoints().boxes(_checkerboard(density=density, border=border), _REFERENCE, look=SLAB,
                                 slice_offset=-9, slice_boxes=SLAB_BOXES)
    colour = {tuple(position): tuple(colour) for position, colour
              in zip(gathered.positions.tolist(), gathered.colours.tolist())}

    def shaded(tint, shade):
        return tuple(int(round(channel * shade)) for channel in tint) + (255,)

    assert colour[(0.5, 0.5, 0.5)] == shaded(SLAB_RED, 1.0)
    assert colour[(0.5, 0.5, 1.5)] == shaded(SLAB_BLUE, 1.0)
    assert colour[(1.5, 0.5, 0.5)] == shaded(SLAB_BLUE, SLAB_DIMMEST)
    assert colour[(1.5, 1.5, 0.5)] == shaded(SLAB_RED, SLAB_DIMMEST)
    lifted = tuple(int(round(channel * SLAB_DIMMEST + (255 - channel * SLAB_DIMMEST) * SLAB_BORDER_LIFT))
                   for channel in SLAB_RED) + (255,)
    assert np.abs(np.subtract(colour[(0.5, 0.5, 2.5)], lifted)).max() <= 1  # float32 rounding at .5
    assert all(lifted[channel] > colour[(1.5, 1.5, 0.5)][channel] for channel in range(3))


def test_a_slab_thinner_than_one_layer_is_refused():
    with pytest.raises(ValueError, match="slice_boxes"):
        BoxPoints().boxes(_checkerboard(), _REFERENCE, look=SLAB, slice_offset=0, slice_boxes=0.5)


class _EvenSource:
    """An 8^3 checkerboard of boxes that never changes."""

    steps = 0

    def reset(self) -> None:
        pass

    def step(self) -> None:
        self.steps += 1

    def view(self):
        return _checkerboard()

    def stats(self) -> dict:
        return {}


def test_the_stepper_toggles_the_slab_moves_it_and_titles_mode_and_place():
    stepper = Stepper(_EvenSource(), paused=True, look=GLOW)
    try:
        _wait(stepper, lambda snapshot: True)
        stepper.toggle_slice()
        stepper.toggle_slab()
        assert stepper.slab and not stepper.slicing
        slab = _wait(stepper, lambda snapshot: snapshot.look == SLAB)
        assert len(slab.positions) == 320 and slab.slice_at == 4.5
        assert "slab z 2.5..6.5" in title_line(slab, 60.0) and "| slab |" in title_line(slab, 60.0)
        stepper.move_slice(1)
        moved = _wait(stepper, lambda snapshot: snapshot.slice_at == 5.5)
        assert "slab z 3.5..7.5" in title_line(moved, 60.0)
        stepper.show(SHOW_ENERGY_VOLUMES)
        _wait(stepper, lambda snapshot: len(snapshot.positions) == 160)
        stepper.toggle_slab()
        whole = _wait(stepper, lambda snapshot: snapshot.look == GLOW)
        assert np.isnan(whole.slice_at) and "slab" not in title_line(whole, 60.0)
        stepper.draw_as(NEAREST)
        stepper.toggle_slab()
        stepper.toggle_slice()
        sliced = _wait(stepper, lambda snapshot: snapshot.look == NEAREST and not np.isnan(snapshot.slice_at))
        assert not stepper.slab and len(sliced.positions) == 32 and "slice z 4.5" in title_line(sliced, 60.0)
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
