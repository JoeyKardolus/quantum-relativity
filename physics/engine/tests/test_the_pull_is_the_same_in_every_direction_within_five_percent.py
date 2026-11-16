"""The pull is direction-free: behind its front it is -G M / r along an axis, a face diagonal and a body diagonal alike, within 5%.

Issue #3 acceptance (direction-free within 5%). A 96-box grid is all joint (-1000 per
box: space that carries the pull and adds none) with a ball of radius 2 boxes of plus
quanta (+1000 per box) at its middle, switched on at step 0. Only `pull_wave` runs
(default settings: light speed 1, step_time 0.25) for 160 steps, 40 time units. For a
source switched on at t = 0 the wave equation's pull inside the light cone is the static
-G M / r (Kirchhoff), so at r = 8..16 boxes, well behind the front at 40 and before the
walls' echo (back at r = 16 only at t = 64), the pull is compared with -G M / r. Each
direction's mean ratio over its lattice points, and every single point, lies within 5% of
1, and the three directions' means within 5% of each other.
"""
import numpy as np

from physics.engine import Settings, State, pull_wave
from physics.engine.tests._gpu import needs_gpu

SIDE = 96
CENTRE = 48
DIRECTIONS = {"axis": (1, 0, 0), "face_diagonal": (1, 1, 0), "body_diagonal": (1, 1, 1)}


@needs_gpu
def test_the_pull_matches_minus_g_m_over_r_along_axes_and_diagonals():
    settings = Settings()
    places = np.arange(SIDE) - CENTRE
    radius = np.sqrt(places[:, None, None] ** 2 + places[None, :, None] ** 2 + places[None, None, :] ** 2)
    quanta = np.where(radius <= 2.0, 1000, -1000).astype(np.int32)
    mass = float(quanta[quanta > 0].sum())
    state = State.from_host(quanta, np.zeros(quanta.shape))
    for _ in range(160):
        pull_wave(state, settings)
    pull = state.pull.numpy()
    means = {}
    for name, direction in DIRECTIONS.items():
        unit = np.asarray(direction)
        ratios = []
        for multiple in range(1, 20):
            offset = multiple * unit
            distance = float(np.linalg.norm(offset))
            if 8.0 <= distance <= 16.0:
                there = tuple(CENTRE + offset)
                ratios.append(pull[there] / (-settings.pull_strength * mass / distance))
        means[name] = float(np.mean(ratios))
        print(f"{name}: pull / (-G M / r) at {len(ratios)} points {np.round(ratios, 4).tolist()}, mean {means[name]:.4f}")
        assert np.all(np.abs(np.asarray(ratios) - 1.0) <= 0.05)
    spread = (max(means.values()) - min(means.values())) / np.mean(list(means.values()))
    print(f"spread of the three means: {spread:.4f}")
    assert spread <= 0.05
