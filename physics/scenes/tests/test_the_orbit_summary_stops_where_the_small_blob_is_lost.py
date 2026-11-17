"""An orbit's numbers cover only the samples in which the small blob could still be followed."""
import math

import numpy as np

from physics.scenes.orbit import _summary


def _circle(samples: int, lost_from: int) -> tuple[list[float], list[dict]]:
    """A blob on a circle of radius 10, a quarter turn in `samples` samples, lost from `lost_from` on."""
    times = [0.25 * index for index in range(samples)]
    measured = []
    for index in range(samples):
        angle = 0.5 * math.pi * index / (samples - 1)
        found = index < lost_from
        measured.append({"offset": np.array([10.0 * math.cos(angle), 10.0 * math.sin(angle), 0.0]) if found else np.zeros(3),
                         "spread": 1.0 if found else 0.0, "small_energy": 100.0 if found else 0.0})
    return times, measured


def test_a_blob_followed_to_the_end_gives_the_whole_quarter_turn():
    times, measured = _circle(41, 41)
    run = _summary(times, measured)
    assert run["followed_time"] == times[-1]
    assert abs(run["turns"] - 0.25) < 1e-9 and abs(run["mean_radius"] - 10.0) < 1e-9


def test_a_blob_lost_half_way_is_summed_up_to_the_loss_only():
    times, measured = _circle(41, 21)
    run = _summary(times, measured)
    assert run["followed_time"] == times[20]
    assert abs(run["turns"] - 0.125) < 1e-9
    assert abs(run["mean_radius"] - 10.0) < 1e-9 and run["min_radius"] > 9.99
    assert len(run["angle"]) == len(run["radius"]) == len(run["angular_momentum"]) == 41
    assert run["radius"][21] == 0.0


def test_a_blob_lost_at_once_has_no_orbit():
    times, measured = _circle(41, 2)
    run = _summary(times, measured)
    assert run["turns"] == 0.0 and math.isinf(run["period"]) and run["mean_radius"] == 0.0


class _Side:
    side = 24


def _tracker_with(fields: list):
    """A tracker whose excess per box comes from `fields`, one per measure() call."""
    from physics.scenes.orbit import _Tracker
    tracker = _Tracker(np.array([6.0, 12.0, 12.0]), 2.0, np.array([18.0, 12.0, 12.0]), 4.0)
    queue = list(fields)
    tracker._excess = lambda universe: queue.pop(0)
    return tracker


def _field(big: float, small: float) -> np.ndarray:
    energy = np.zeros((24, 24, 24))
    energy[6, 12, 12], energy[18, 12, 12] = big, small
    return energy


def test_the_small_blob_is_lost_for_good_once_its_place_holds_over_twice_its_start():
    tracker = _tracker_with([_field(1000.0, 10.0), _field(1000.0, 19.0), _field(1000.0, 21.0), _field(1000.0, 10.0)])
    held = [tracker.measure(_Side())["small_energy"] for _ in range(4)]
    assert held == [10.0, 19.0, 0.0, 0.0]


def test_the_small_blob_is_lost_when_nothing_is_left_at_its_place():
    tracker = _tracker_with([_field(1000.0, 10.0), _field(1000.0, 0.0), _field(1000.0, 10.0)])
    assert [tracker.measure(_Side())["small_energy"] for _ in range(3)] == [10.0, 0.0, 0.0]
