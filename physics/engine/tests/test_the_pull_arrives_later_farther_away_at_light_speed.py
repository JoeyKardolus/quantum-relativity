"""The pull spreads box to box at light speed: a source's pull arrives later the farther away, at the set speed.

Issue #3: pull spreads at a fixed finite speed instead of being solved across all space
at once. A grid of 96 boxes a side is all joint (-1000 per box: space that carries the
pull and adds none) with a ball of radius 2 boxes of plus quanta (+1000 per box) at its
middle, switched on at step 0. Only `pull_wave` runs (default settings: light speed 1
box per unit time, step_time 0.25). At distances r = 10, 14, ..., 34 boxes along +x the
arrival is the first step at which the pull there reaches half its static value
-G M / r. The arrival times rise with r, and the slope of a straight-line fit of r
against arrival time is the light speed within 10%. The walls (48 boxes out) reflect,
but their echo reaches r = 34 only after its arrival. The same is measured along the
body diagonal and printed.
"""
import numpy as np

from physics.engine import Settings, State, pull_wave
from physics.engine.tests._gpu import needs_gpu

SIDE = 96
CENTRE = 48
DISTANCES = np.arange(10, 35, 4)


@needs_gpu
def test_the_pull_arrives_at_r_over_c():
    settings = Settings()
    places = np.arange(SIDE) - CENTRE
    radius = np.sqrt(places[:, None, None] ** 2 + places[None, :, None] ** 2 + places[None, None, :] ** 2)
    quanta = np.where(radius <= 2.0, 1000, -1000).astype(np.int32)
    mass = float(quanta[quanta > 0].sum())
    state = State.from_host(quanta, np.zeros(quanta.shape))
    diagonal = np.rint(DISTANCES / np.sqrt(3.0)).astype(int)
    axis_arrival = np.full(len(DISTANCES), np.nan)
    diagonal_arrival = np.full(len(DISTANCES), np.nan)
    for count in range(1, 260):
        pull_wave(state, settings)
        pull = state.pull.numpy()
        time = count * settings.step_time
        for index, distance in enumerate(DISTANCES):
            if np.isnan(axis_arrival[index]) and pull[CENTRE + distance, CENTRE, CENTRE] <= -0.5 * settings.pull_strength * mass / distance:
                axis_arrival[index] = time
            corner = CENTRE + diagonal[index]
            reach = np.sqrt(3.0) * diagonal[index]
            if np.isnan(diagonal_arrival[index]) and pull[corner, corner, corner] <= -0.5 * settings.pull_strength * mass / reach:
                diagonal_arrival[index] = time
    speed = np.polyfit(axis_arrival, DISTANCES, 1)[0]
    diagonal_speed = np.polyfit(diagonal_arrival, np.sqrt(3.0) * diagonal, 1)[0]
    print(f"axis arrivals {axis_arrival.tolist()} for r {DISTANCES.tolist()}: speed {speed:.4f} "
          f"(set {settings.light_speed}); body diagonal speed {diagonal_speed:.4f}")
    assert not np.isnan(axis_arrival).any()
    assert np.all(np.diff(axis_arrival) > 0)
    assert abs(speed / settings.light_speed - 1.0) <= 0.10
