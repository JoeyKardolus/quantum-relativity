"""Axiom 07: a volume only pulls, a joint only pushes; empty space does nothing.

Audit answers (owner, 2026-11-16): there is no size at rest; an energy volume only ever
pulls and a joint only ever pushes, and empty space does nothing. Measured four ways:
  1. default universe (seed 1, Settings(), axiom_run.py), every 10 steps for 300 steps:
     the engine's own cell push (cell_laws.py, read by axiom_pressure.py, the pull not
     in it) is 0 for every energy volume (a volume pushes nothing), at least 0 for every
     joint (a joint never pulls) and 0 for every id that is no cell (empty space)
  2. the energy a box's fluid feels (box_potential.py) with the pull 0: a plus box of
     1000 quanta feels 0 (it has no push of its own), an empty box 0 (a joint box feels
     its joint's push, here Settings().push)
  3. staged: a cube of joint boxes only (-1000 each, no plus anywhere), 100 steps of the
     pull wave: the pull stays exactly 0 (a joint adds no pull)
  4. staged: a universe of empty boxes only, 100 steps: the pull stays 0
"""
import numpy as np
import warp as wp

from physics.engine import Settings, State, box_potential, pull_wave
from physics.engine.tests._gpu import needs_gpu


@wp.kernel
def _feel(amounts: wp.array(dtype=wp.int32), push: float, felt: wp.array(dtype=wp.float32)):
    index = wp.tid()
    felt[index] = box_potential(amounts[index], 0.0, push)


def _pull_after(quanta: np.ndarray, steps: int) -> float:
    """The largest |pull| after `steps` pull-wave steps on a fixed start."""
    state = State.from_host(quanta, np.zeros(quanta.shape))
    settings = Settings()
    for _ in range(steps):
        pull_wave(state, settings)
    return float(np.abs(state.pull.numpy()).max())


@needs_gpu
def test_volumes_push_nothing_joints_never_pull_and_empty_space_does_nothing():
    from physics.engine.tests.axiom_run import axiom_run
    run = axiom_run()
    kinds = run["kinds"]
    volume_push = max(float(check["pressure"][kinds == 1].max()) for check in run["checks"])
    joint_least = min(float(check["pressure"][kinds == -1].min()) for check in run["checks"])
    ring_most = max(float(np.abs(check["pressure"][kinds == 0]).max()) for check in run["checks"])
    settings = Settings()
    amounts = wp.array(np.array([1000, 0, -1000], dtype=np.int32), dtype=wp.int32)
    felt = wp.zeros(3, dtype=wp.float32)
    wp.launch(_feel, dim=3, inputs=[amounts, settings.push, felt])
    plus_feels, empty_feels, joint_feels = (float(value) for value in felt.numpy())
    joints_only = np.zeros((32,) * 3, dtype=np.int32)
    joints_only[4:28, 4:28, 4:28] = -1000
    joint_pull = _pull_after(joints_only, 100)
    empty_pull = _pull_after(np.zeros((16,) * 3, dtype=np.int32), 100)
    print(f"default run, every 10 steps: largest energy-volume push {volume_push:.4g}, least joint push "
          f"{joint_least:.4g}, largest push of an id that is no cell {ring_most:.4g}; a plus box of 1000 quanta feels {plus_feels:.4g} "
          f"of its own push (pull 0), a joint box {joint_feels:.4g}, an empty box {empty_feels:.4g}; pull made by "
          f"joints only {joint_pull:.3g}, by empty space {empty_pull:.3g}")
    assert ring_most == 0.0 and empty_feels == 0.0 and empty_pull == 0.0
    assert joint_least >= 0.0 and joint_pull == 0.0
    assert volume_push == 0.0 and plus_feels == 0.0
