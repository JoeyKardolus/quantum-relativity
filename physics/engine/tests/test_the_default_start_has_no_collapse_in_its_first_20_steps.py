"""The default start holds no joint past the seal, and no joint collapses in its first 20 steps.

Bible foundation: the start is an actual grid of energy volumes wrapped in joints, the
unevenness in slightly different energies, and one correction lets sizes follow energy,
so no joint starts squeezed. The default start (new_universe(), seed 1: 7**3 cells of
10**3 boxes) has every joint box below the seal squeeze, and 20 steps of the default
settings collapse no joint box.
"""
import numpy as np

from physics.engine import Settings, new_universe, step
from physics.engine.tests._gpu import needs_gpu


@needs_gpu
def test_no_joint_starts_sealed_and_none_collapses_in_20_steps():
    settings = Settings()
    state = new_universe()
    quanta = state.quanta.numpy()
    start_squeeze = float((-quanta[quanta < 0]).max() / settings.joint_rest - 1.0)
    for _ in range(20):
        step(state, settings)
    quanta = state.quanta.numpy()
    squeeze = float((-quanta[quanta < 0]).max() / settings.joint_rest - 1.0)
    collapsed = int(state.tallies.numpy()[0, 1])
    print(f"squeeze max at the start {start_squeeze:.3f}, after 20 steps {squeeze:.3f}, collapsed {collapsed}")
    assert start_squeeze < settings.seal_squeeze
    assert collapsed == 0
